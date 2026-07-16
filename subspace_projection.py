from utils.segment_splitters import *
import argparse
import os
import json
from tqdm import tqdm
from functools import partial
from collections import defaultdict
import random
import numpy as np
import string
import re
import math
from jinja2.exceptions import TemplateError
import bisect

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
except ModuleNotFoundError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    DataCollatorForSeq2Seq = None

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None

from utils.generation_utils import *


class DataCollatorForSeq2SeqWithOffsets:
    def __init__(self, tokenizer, pad_to_multiple_of=8, padding=True):
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            padding=padding,
        )

    def __call__(self, features):
        offset_mappings = []
        model_features = []
        for feature in features:
            feature = dict(feature)
            offset_mappings.append(feature.pop("offset_mapping"))
            model_features.append(feature)

        batch = self.base_collator(model_features)
        seq_len = batch["input_ids"].size(1)

        padded_offsets = []
        for offsets in offset_mappings:
            offsets = torch.as_tensor(offsets, dtype=torch.long)
            if offsets.ndim == 1:
                offsets = offsets.reshape(-1, 2)

            if offsets.size(0) > seq_len:
                offsets = offsets[:seq_len]

            pad_len = seq_len - offsets.size(0)
            if pad_len > 0:
                pad = offsets.new_full((pad_len, 2), -1)
                if self.tokenizer.padding_side == "left":
                    offsets = torch.cat([pad, offsets], dim=0)
                else:
                    offsets = torch.cat([offsets, pad], dim=0)

            padded_offsets.append(offsets)

        batch["offset_mapping"] = torch.stack(padded_offsets, dim=0)
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--train_file", type=str, required=False)
    parser.add_argument("--scoring_mode", type=str, default="reps", choices=["reps", "birep"])
    parser.add_argument("--reps_output_mode", type=str, default="both", choices=["token", "sample", "both"])
    parser.add_argument("--unsafe_anchor_file", "--unsafe_file", dest="unsafe_anchor_file", type=str,
                        default="")
    parser.add_argument("--safe_anchor_file", "--safe_file", dest="safe_anchor_file", type=str,
                        default="")
    parser.add_argument("--rep_layer_start", type=int, default=None)
    parser.add_argument("--rep_layer_end", type=int, default=None)
    parser.add_argument("--rep_layers", type=str, default=None)
    parser.add_argument("--layer_agg", type=str, default="cat", choices=["mean", "max", "cat"])
    parser.add_argument("--birep_top_avg", type=int, default=100)
    parser.add_argument("--device_map", type=str, default="auto", choices=["auto", "cpu"])
    parser.add_argument("--output_json", type=str, default="results/safety_mask.json")
    parser.add_argument("--span_granularity", type=str, default="word", choices=["token","word"])
    parser.add_argument("--word_agg", type=str, default="mean", choices=["sum","mean","max"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--subspace_components", type=int, default=16)
    parser.add_argument("--subspace_score_mode", type=str, default="proj_energy", choices=["proj_energy"])
    parser.add_argument("--anchor_batch_size", type=int, default=16)
    parser.add_argument("--subspace_cache_dir", type=str, default="results/subspace_cache")
    parser.add_argument("--no_subspace_cache", action="store_true")
    parser.add_argument("--splitter_type", type=str, default="regex", choices=["regex", "jieba", "nltk", "spacy"])
    args = parser.parse_args()

    splitter_map = {
        "regex": RegexSegmentSplitter,
        "jieba": JiebaSegmentSplitter,
        "nltk": NltkSegmentSplitter,
        "spacy": SpacySegmentSplitter
    }
    segment_splitter = splitter_map[args.splitter_type]()
    print(f"*** Using Text Splitter: {args.splitter_type} ***")

    if args.scoring_mode == 'birep':
        args.scoring_mode = 'reps'

    if args.scoring_mode == 'reps':
        resolved_rep_layers, layer_source = resolve_rep_layers(
            args.model_path,
            rep_layer_start=args.rep_layer_start,
            rep_layer_end=args.rep_layer_end,
            rep_layers=args.rep_layers,
        )
        args.rep_layers = resolved_rep_layers
        print(f"[reps] Using safety critical layers: {args.rep_layers}")

    if args.max_seq_length % 8 != 0:
        args.max_seq_length = ((args.max_seq_length + 7) // 8) * 8

    if not all([args.model_path, args.train_file]):
        raise SystemExit("Missing required arguments.")
        
    if torch is None or AutoModelForCausalLM is None or load_dataset is None:
        raise SystemExit("Missing dependencies.")

    if args.output_json == "results/safety_mask.json":
        model_name = os.path.basename(os.path.normpath(args.model_path))
        dataset_name = os.path.splitext(os.path.basename(args.train_file))[0]
        output_dir = os.path.join("results", model_name)
        os.makedirs(output_dir, exist_ok=True)
        args.output_json = os.path.join(output_dir, f"{dataset_name}_safety_spans.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    output_base, output_ext = os.path.splitext(args.output_json)
    if output_ext.lower() != ".json":
        output_ext = ".json"
    word_level_path = f"{output_base}_word_level{output_ext}"
    sample_level_path = f"{output_base}_sample_level{output_ext}"
    preview_path = f"{output_base}_preview{output_ext}"

    set_seed(args.seed)

    if args.device_map == "auto" and not torch.cuda.is_available():
        args.device_map = "cpu"

    print(f"Loading Base Model: {args.model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()

    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device_map == "cpu" else next(base_model.parameters()).device
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    special_token_ids = set(tokenizer.all_special_ids)

    print("Loading anchors and building uncentered subspaces...")
    unsafe_subspace, safe_subspace = build_anchor_subspace_subspaces(
        base_model,
        tokenizer,
        args.unsafe_anchor_file,
        args.safe_anchor_file,
        rep_layers=args.rep_layers,
        layer_agg=args.layer_agg,
        model_input_device=compute_device,
        subspace_components=args.subspace_components,
        anchor_batch_size=args.anchor_batch_size,
        cache_dir=None if args.no_subspace_cache else args.subspace_cache_dir,
        model_cache_id=args.model_path,
    )

    common_config = {
        "scoring_mode": "uncentered_subspace_diff",
        "config": {
            "rep_layers": args.rep_layers,
            "layer_agg": args.layer_agg,
            "subspace_components": args.subspace_components,
            "span_granularity": args.span_granularity,
            "word_agg": args.word_agg,
            "splitter_type": args.splitter_type,
        },
    }

    print(f"Loading Data: {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file)['train']
    dataset = dataset.map(lambda x, i: {"idx": i}, with_indices=True)
    original_dataset = dataset
    
    process_func = partial(process_and_tokenize, tokenizer=tokenizer, max_length=args.max_seq_length)
    cols_to_keep = ["idx", "response_start_index", "original_response"]
    cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
    processed = dataset.map(process_func, batched=False, remove_columns=cols_to_remove)
    tensor_columns = ["input_ids", "labels", "attention_mask", "idx", "prompt_length_tokens"]
    if args.reps_output_mode != "sample":
        tensor_columns.append("offset_mapping")
    processed.set_format(type="pt", columns=tensor_columns)
    
    if args.reps_output_mode == "sample":
        collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True)
    else:
        collator = DataCollatorForSeq2SeqWithOffsets(tokenizer, pad_to_multiple_of=8, padding=True)
    dataloader = torch.utils.data.DataLoader(processed, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    results_by_sample = defaultdict(list)
    preview_list = []
    sample_preview = []

    sample_writer = None
    sample_writer_first = True
    if args.reps_output_mode in ["sample", "both"]:
        sample_writer = open(sample_level_path, 'w', encoding='utf-8')
        sample_writer.write('{\n')
        sample_writer.write(f'  "scoring_mode": {json.dumps(common_config["scoring_mode"], ensure_ascii=False)},\n')
        sample_writer.write('  "config": ')
        json.dump(common_config["config"], sample_writer, ensure_ascii=False, indent=2)
        sample_writer.write(',\n')
        sample_writer.write('  "level": "sample",\n')
        sample_writer.write('  "data": [\n')

    print("*** Scanning Tokens & Extracting Spans (Using REPS) ***")

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(compute_device)
            labels = batch['labels'].to(compute_device)
            attn_mask = batch['attention_mask'].to(compute_device)
            offset_mapping = batch.get('offset_mapping')
            indices = batch['idx']
            batch_original_responses = [processed['original_response'][int(idx)] for idx in indices]
            batch_original_rows = [original_dataset[int(idx)] for idx in indices]
            curr_bs = input_ids.size(0)

            if args.reps_output_mode != "sample":
                batch_res_starts = [processed['response_start_index'][int(idx)] for idx in indices]
                batch_original_outputs = [original_dataset[int(idx)]['output'] for idx in indices]

            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            if args.reps_output_mode == "sample":
                token_derivs = None
                sample_scores_batch, sample_last_token_pos = compute_subspace_sample_scores(
                    outputs.hidden_states,
                    labels,
                    unsafe_subspace,
                    safe_subspace,
                    rep_layers=args.rep_layers,
                    layer_agg=args.layer_agg,
                )
            else:
                token_derivs = compute_subspace_token_scores(
                    outputs.hidden_states,
                    labels,
                    unsafe_subspace,
                    safe_subspace,
                    rep_layers=args.rep_layers,
                    layer_agg=args.layer_agg,
                )
                sample_scores_batch, sample_last_token_pos = compute_sample_scores_from_token_scores(token_derivs, labels)

            for b_idx in range(curr_bs):
                sample_idx = int(indices[b_idx].item())
                last_pos = sample_last_token_pos[b_idx]
                original_row = dict(batch_original_rows[b_idx])
                original_row.pop("idx", None)
                stable_id = get_sample_id(original_row, sample_idx)
                original_row["id"] = stable_id
                
                original_row["text"] = build_export_text(original_row, fallback_text=batch_original_responses[b_idx])

                sim_score = float(sample_scores_batch[b_idx]) if last_pos >= 0 else float('nan')
                original_row["sim_score"] = sim_score
                
                if sample_writer is not None and not np.isnan(sim_score):
                    if not sample_writer_first: sample_writer.write(',\n')
                    sample_writer.write('    ')
                    sample_writer.write(json.dumps(original_row, ensure_ascii=False))
                    sample_writer_first = False
                    if len(sample_preview) < 5000:
                        sample_preview.append(original_row)

                if args.reps_output_mode == "sample":
                    continue

                cur_offsets = offset_mapping[b_idx]
                pure_output_text = batch_original_outputs[b_idx]
                res_start = batch_res_starts[b_idx]

                valid = (labels[b_idx] != -100)
                if not valid.any(): continue

                valid_indices = torch.nonzero(valid).squeeze(-1)
                
                b_derivs = token_derivs[b_idx][valid]

                token_entries = []
                prev_score = 0.0
                
                for i, t_idx in enumerate(valid_indices):
                    token_pos = int(t_idx.item())
                    token_id_val = input_ids[b_idx, token_pos].item()
                    if token_id_val in special_token_ids: continue

                    start_char, end_char = cur_offsets[token_pos].tolist()
                    if start_char == -1 or end_char == -1 or start_char >= end_char: continue

                    rel_start = start_char - res_start
                    rel_end = end_char - res_start

                    if rel_start < 0: continue 

                    try:
                        token_text = pure_output_text[rel_start:rel_end]
                    except IndexError: 
                        continue

                    if not is_meaningful_span(token_text): continue

                    cur_abs = float(b_derivs[i].item())
                    delta = cur_abs - prev_score
                    prev_score = cur_abs 

                    token_entries.append({
                        "span": [rel_start, rel_end], 
                        "text": token_text,
                        "score": delta,
                    })

                if args.span_granularity == "word":
                    merged = merge_token_entries_to_english_words(token_entries, pure_output_text, segment_splitter, agg=args.word_agg)
                    for w in merged:
                        score_val = float('nan') if math.isnan(w["score"]) else round(w["score"], 5)
                        results_by_sample[str(stable_id)].append({
                            "span": w["span"],
                            "text": w["text"],
                            "score": score_val, 
                        })
                else:
                    for te in token_entries:
                        results_by_sample[str(stable_id)].append({
                            "span": te["span"],
                            "text": te["text"],
                            "score": round(te["score"], 5), 
                        })

    if sample_writer is not None:
        if not sample_writer_first:
            sample_writer.write('\n')
        sample_writer.write('  ]\n')
        sample_writer.write('}\n')
        sample_writer.close()
        print(f"*** Sample-level saved to {sample_level_path} ***")

    print("*** Saving Outputs (No Global Ranking) ***")

    if args.reps_output_mode in ["token", "both"]:
        word_level_payload = {
            **common_config,
            "level": "word" if args.span_granularity == "word" else "token",
            "data": results_by_sample,
        }
        with open(word_level_path, 'w', encoding='utf-8') as f:
            json.dump(word_level_payload, f, indent=2, ensure_ascii=False)
        print(f"*** Word-level saved to {word_level_path} ***")

    with open(preview_path, 'w', encoding='utf-8') as f:
        json.dump({"word_preview": preview_list, "sample_preview": sample_preview}, f, indent=4, ensure_ascii=False)
    print(f"*** Preview saved to {preview_path} ***")

if __name__ == "__main__":
    main()
