from utils.segment_splitters import *
import argparse
import hashlib
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"*** Random Seed Set to {seed} ***")

def parse_rep_layers_arg(rep_layers):
    if rep_layers is None:
        return None
    if isinstance(rep_layers, (list, tuple)):
        return [int(x) for x in rep_layers]
    rep_layers = str(rep_layers).strip()
    if not rep_layers:
        return None
    layers = []
    for part in rep_layers.split(','):
        part = part.strip()
        if not part:
            continue
        layers.append(int(part))
    return layers or None

def resolve_rep_layers(model_path, rep_layer_start=None, rep_layer_end=None, rep_layers=None):
    model_name = os.path.basename(os.path.normpath(model_path))

    default_layers = [14]

    explicit_layers = parse_rep_layers_arg(rep_layers)
    if explicit_layers is not None:
        resolved_layers = explicit_layers
        source = "arg_rep_layers"
    elif rep_layer_start is not None or rep_layer_end is not None:
        default_start = default_layers[0]
        default_end = default_layers[-1]
        start = int(rep_layer_start) if rep_layer_start is not None else default_start
        end = int(rep_layer_end) if rep_layer_end is not None else default_end
        step = 1 if end >= start else -1
        resolved_layers = list(range(start, end + step, step))
        source = "arg_rep_layer_range"
    else:
        resolved_layers = default_layers
        source = "default"

    return resolved_layers, source

def apply_chat_template_safe(tokenizer, messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as e:
        msg = str(e)
        if isinstance(e, TemplateError) and ("System role not supported" in msg or "system role" in msg.lower()):
            sys_text = ""
            rest = messages
            if rest and rest[0].get("role") == "system":
                sys_text = rest[0].get("content", "")
                rest = rest[1:]

            new_msgs = []
            injected = False
            for m in rest:
                if (not injected) and m.get("role") == "user":
                    m = dict(m)
                    if sys_text:
                        m["content"] = f"{sys_text}\n\n{m.get('content','')}"
                    injected = True
                    new_msgs.append(m)
                else:
                    new_msgs.append(m)
            return tokenizer.apply_chat_template(new_msgs, **kwargs)
        raise

def compute_shifted_token_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    token_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_loss = token_loss.view(shift_labels.size())
    return F.pad(token_loss, (1, 0), value=0.0)

def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def get_sample_id(original_row, fallback_idx):
    if isinstance(original_row, dict) and "id" in original_row and original_row["id"] is not None:
        return original_row["id"]
    return int(fallback_idx)

def aggregate_layers(hidden_states, rep_layers, layer_agg='mean'):
    if rep_layers is None or len(rep_layers) == 0:
        raise ValueError("rep_layers must contain at least one layer index")

    max_layer_idx = len(hidden_states) - 1
    selected = []
    for layer_idx in rep_layers:
        idx = int(layer_idx)
        if idx < 0:
            idx = len(hidden_states) + idx
        if idx < 0 or idx > max_layer_idx:
            raise ValueError(f"Layer index {layer_idx} is out of range [0, {max_layer_idx}]")
        selected.append(hidden_states[idx])

    if len(selected) == 1:
        return selected[0]

    if layer_agg == 'mean':
        return torch.stack(selected, dim=0).mean(dim=0)
    if layer_agg == 'max':
        return torch.stack(selected, dim=0).max(dim=0).values
    if layer_agg == 'cat':
        normed = []
        for h in selected:
            h = h.float()
            h_normed = F.normalize(h, p=2, dim=-1)
            normed.append(h_normed)
        
        cat_h = torch.cat(normed, dim=-1)
        return F.normalize(cat_h, p=2, dim=-1)
        
    raise ValueError(f"Unknown layer_agg={layer_agg}")


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _anchor_cache_metadata(
    model_cache_id,
    unsafe_anchor_file,
    safe_anchor_file,
    rep_layers,
    layer_agg,
    subspace_components,
):
    def file_meta(path):
        abs_path = os.path.abspath(path)
        stat = os.stat(abs_path)
        return {
            "path": abs_path,
            "size": stat.st_size,
            "sha256": _file_sha256(abs_path),
        }

    return {
        "version": 3,
        "model_cache_id": os.path.abspath(model_cache_id) if model_cache_id else None,
        "unsafe_anchor_file": file_meta(unsafe_anchor_file),
        "safe_anchor_file": file_meta(safe_anchor_file),
        "rep_layers": [int(x) for x in rep_layers],
        "layer_agg": layer_agg,
        "subspace_components": int(subspace_components),
    }


def _anchor_cache_path(cache_dir, metadata):
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    model_name = os.path.basename(os.path.normpath(metadata.get("model_cache_id") or "model"))
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return os.path.join(cache_dir, f"{safe_model_name}_{key}.pt")


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def build_anchor_subspace_subspaces(
    model,
    tokenizer,
    unsafe_anchor_file,
    safe_anchor_file,
    rep_layers,
    layer_agg='mean',
    model_input_device=None,
    subspace_components=8,
    anchor_batch_size=16,
    cache_dir=None,
    model_cache_id=None,
):
    anchor_batch_size = max(1, int(anchor_batch_size))
    cache_path = None
    cache_metadata = None
    if cache_dir:
        cache_metadata = _anchor_cache_metadata(
            model_cache_id,
            unsafe_anchor_file,
            safe_anchor_file,
            rep_layers,
            layer_agg,
            subspace_components,
        )
        cache_path = _anchor_cache_path(cache_dir, cache_metadata)
        if os.path.exists(cache_path):
            payload = _torch_load_cpu(cache_path)
            if payload.get("metadata") == cache_metadata:
                print(f"Loading cached anchor subspaces: {cache_path}")
                return payload["unsafe_subspace"], payload["safe_subspace"]

    unsafe_data = load_jsonl(unsafe_anchor_file)
    safe_data = load_jsonl(safe_anchor_file)

    def _extract_last_token_rep(data_rows, desc):
        reps = []
        items = [row.get('messages', None) for row in data_rows if row.get('messages', None)]
        if not items:
            raise ValueError(f"No valid anchor messages found for {desc}")

        with torch.inference_mode():
            for start in tqdm(range(0, len(items), anchor_batch_size), desc=desc):
                batch_messages = items[start:start + anchor_batch_size]
                if model_input_device is None:
                    model_input_device_local = next(model.parameters()).device
                else:
                    model_input_device_local = model_input_device

                texts = [
                    apply_chat_template_safe(tokenizer, messages, tokenize=False)
                    for messages in batch_messages
                ]
                enc = tokenizer(
                    texts,
                    return_tensors='pt',
                    padding=True,
                    add_special_tokens=False,
                )
                input_ids = enc["input_ids"].to(model_input_device_local)
                attention_mask = enc["attention_mask"].to(model_input_device_local)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = aggregate_layers(outputs.hidden_states, rep_layers, layer_agg=layer_agg)
                if tokenizer.padding_side == "left":
                    last_indices = torch.full(
                        (input_ids.size(0),),
                        input_ids.size(1) - 1,
                        device=hidden.device,
                        dtype=torch.long,
                    )
                else:
                    last_indices = attention_mask.sum(dim=1).to(hidden.device, dtype=torch.long) - 1
                batch_indices = torch.arange(input_ids.size(0), device=hidden.device)
                reps.append(hidden[batch_indices, last_indices].detach().cpu().float())

        if not reps:
            raise ValueError(f"No valid anchor reps extracted for {desc}")

        reps = torch.cat(reps, dim=0)   # [n, d]
        return reps

    def _fit_uncentered_subspace_basis(reps, k, desc):
        n, d = reps.shape
        if n < 1:
            raise ValueError(f"{desc}: empty reps")

        _, singular_values, vh = torch.linalg.svd(reps, full_matrices=False)

        k = min(int(k), vh.size(0), d)
        if k <= 0:
            raise ValueError(f"subspace_components must be >=1, got {k}")

        basis = vh[:k].transpose(0, 1).contiguous()
        evals = singular_values[:k].pow(2) / float(n)

        return {
            "basis": basis,
            "evals": evals,
            "reps": reps,
        }

    unsafe_reps = _extract_last_token_rep(unsafe_data, 'unsafe_anchors')
    safe_reps = _extract_last_token_rep(safe_data, 'safe_anchors')

    unsafe_subspace = _fit_uncentered_subspace_basis(unsafe_reps, subspace_components, 'unsafe')
    safe_subspace = _fit_uncentered_subspace_basis(safe_reps, subspace_components, 'safe')

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(
            {
                "metadata": cache_metadata,
                "unsafe_subspace": unsafe_subspace,
                "safe_subspace": safe_subspace,
            },
            cache_path,
        )
        print(f"Saved anchor subspaces cache: {cache_path}")

    return unsafe_subspace, safe_subspace


def compute_subspace_scores_for_reps(rep, unsafe_subspace, safe_subspace, eps=1e-12):
    unsafe_basis = unsafe_subspace["basis"].to(rep.device, dtype=rep.dtype)
    safe_basis = safe_subspace["basis"].to(rep.device, dtype=rep.dtype)

    rep_norm = torch.norm(rep, p=2, dim=-1).clamp_min(eps)
    unsafe_proj_norm = torch.norm(rep @ unsafe_basis, p=2, dim=-1)
    safe_proj_norm = torch.norm(rep @ safe_basis, p=2, dim=-1)
    return unsafe_proj_norm / rep_norm - safe_proj_norm / rep_norm

def compute_subspace_token_scores(
    hidden_states,
    labels,
    unsafe_subspace,
    safe_subspace,
    rep_layers,
    layer_agg='mean',
    eps=1e-12,
):
    token_rep = aggregate_layers(hidden_states, rep_layers, layer_agg=layer_agg)
    bsz, seq_len, dim = token_rep.shape
    flat_rep = token_rep.reshape(-1, dim)
    token_scores = compute_subspace_scores_for_reps(
        flat_rep,
        unsafe_subspace,
        safe_subspace,
        eps=eps,
    ).view(bsz, seq_len)
    return token_scores


def compute_subspace_sample_scores(
    hidden_states,
    labels,
    unsafe_subspace,
    safe_subspace,
    rep_layers,
    layer_agg='mean',
    eps=1e-12,
):
    token_rep = aggregate_layers(hidden_states, rep_layers, layer_agg=layer_agg)
    valid_mask = labels != -100
    positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    last_indices_t = positions.masked_fill(~valid_mask, -1).max(dim=1).values

    sample_scores = [float('nan')] * labels.size(0)
    last_indices = [int(x.item()) for x in last_indices_t]
    valid_samples = last_indices_t >= 0

    if valid_samples.any():
        batch_indices = torch.nonzero(valid_samples).squeeze(-1)
        reps = token_rep[batch_indices, last_indices_t[valid_samples]]
        scores = compute_subspace_scores_for_reps(
            reps,
            unsafe_subspace,
            safe_subspace,
            eps=eps,
        )
        for sample_idx, score in zip(batch_indices.tolist(), scores.tolist()):
            sample_scores[sample_idx] = float(score)

    return sample_scores, last_indices

def compute_sample_scores_from_token_scores(token_scores, labels):
    sample_scores = []
    last_indices = []
    for b in range(token_scores.size(0)):
        valid_pos = torch.nonzero(labels[b] != -100).squeeze(-1)
        if valid_pos.numel() == 0:
            sample_scores.append(float('nan'))
            last_indices.append(-1)
            continue
        last_idx = int(valid_pos[-1].item())
        sample_scores.append(float(token_scores[b, last_idx].item()))
        last_indices.append(last_idx)
    return sample_scores, last_indices

def _maybe_fix_metaspace_offsets(tokenizer, text: str, input_ids, offsets):
    try:
        toks = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    except Exception:
        return offsets

    if not any(t.startswith("Ġ") for t in toks):
        return offsets

    fixed = []
    text_len = len(text)
    for tok, (s, e) in zip(toks, offsets.tolist()):
        s = int(s); e = int(e)
        if s < 0 or e < 0 or s >= e or s >= text_len:
            fixed.append((s, e))
            continue

        if tok.startswith("Ġ"):
            while s < e and s < text_len and text[s].isspace():
                s += 1
        fixed.append((s, e))
    return torch.tensor(fixed, dtype=offsets.dtype)

def process_and_tokenize(example, tokenizer, max_length):
    raw_response = example["output"]
    prompt = example["instruction"]
    if example.get("input", ""):
        prompt += "\n" + example["input"]

    has_template = getattr(tokenizer, "chat_template", None)
    system_prompt = "You are a helpful assistant."

    if has_template:
        messages_full = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw_response},
        ]
        messages_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        full_text = apply_chat_template_safe(tokenizer, messages_full, tokenize=False, add_generation_prompt=False)
        prompt_text = apply_chat_template_safe(tokenizer, messages_prompt, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        full_text = f"{prompt_text} {raw_response}"

    start_hint = max(0, len(prompt_text) - 20)
    response_start_index = full_text.find(raw_response, start_hint)
    
    if response_start_index == -1:
        response_start_index = full_text.find(raw_response)
        
    if response_start_index == -1:
        stripped_res = raw_response.strip()
        if stripped_res:
            idx = full_text.find(stripped_res, start_hint)
            if idx == -1:
                idx = full_text.find(stripped_res)
            if idx != -1:
                offset = raw_response.find(stripped_res)
                response_start_index = max(0, idx - offset)
                
    if response_start_index == -1:
        response_start_index = max(0, len(full_text) - len(raw_response))

    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    input_ids = enc.input_ids[0]
    attention_mask = enc.attention_mask[0]
    offsets = enc.offset_mapping[0]
    offsets = _maybe_fix_metaspace_offsets(tokenizer, full_text, input_ids, offsets)

    labels = input_ids.clone()
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False).input_ids[0]
    cut = min(prompt_ids.size(0), labels.size(0))
    labels[:cut] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "offset_mapping": offsets,
        "response_start_index": response_start_index,
        "original_response": raw_response,
        "prompt_length_tokens": cut, 
    }

def is_meaningful_span(token_text):
    if not token_text: return False
    stripped = token_text.strip()
    if not stripped: return False
    if stripped in {'▁', '\u2581'}: return False
    if all(char in string.punctuation for char in stripped): return False
    return True

SPECIAL_TOKEN_RE = re.compile(r"<\|[^<>|]+\|>")

def sanitize_export_text(text):
    if text is None:
        return ""
    text = str(text)
    text = SPECIAL_TOKEN_RE.sub("", text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_export_text(original_row, fallback_text=""):
    instruction = str(original_row.get("instruction", "") or "")
    input_text = str(original_row.get("input", "") or "")
    output_text = str(original_row.get("output", "") or "")

    if instruction or input_text or output_text:
        if input_text.strip():
            text = (
                "Below is an instruction that describes a task, paired with an input that provides "
                "further context. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output_text}"
            )
        else:
            text = (
                "Below is an instruction that describes a task. Write a response that appropriately "
                "completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output_text}"
            )
        return sanitize_export_text(text)

    existing_text = original_row.get("text", "")
    if isinstance(existing_text, str) and existing_text.strip():
        return sanitize_export_text(existing_text)

    return sanitize_export_text(fallback_text)


def _best_overlapping_word(spans, starts, token_start: int, token_end: int):
    if not spans: return None
    i = bisect.bisect_right(starts, token_start) - 1
    candidates = []
    for j in (i, i + 1):
        if 0 <= j < len(spans):
            ws, we = spans[j]
            overlap = min(token_end, we) - max(token_start, ws)
            if overlap > 0: candidates.append((overlap, ws, we))
    
    if not candidates:
        k = bisect.bisect_left(starts, token_end) - 1
        for j in (k, k + 1):
            if 0 <= j < len(spans):
                ws, we = spans[j]
                overlap = min(token_end, we) - max(token_start, ws)
                if overlap > 0: candidates.append((overlap, ws, we))

    if not candidates: return None
    candidates.sort(reverse=True)
    _, ws, we = candidates[0]
    return ws, we

def merge_token_entries_to_english_words(token_entries, raw_text: str, segment_splitter: BaseSegmentSplitter, agg: str = "sum"):
    if not raw_text: 
        return []
        
    word_spans, word_starts = segment_splitter.get_spans(raw_text)

    buckets = {}
    for ws, we in word_spans:
        buckets[(ws, we)] = {
            "scores": [],
            "loss": []
        }

    if token_entries:
        for te in token_entries:
            s, e = te["span"]
            w = _best_overlapping_word(word_spans, word_starts, int(s), int(e))
            if w is None: continue
            ws, we = w
            buckets[(ws, we)]["scores"].append(float(te["score"]))
            buckets[(ws, we)]["loss"].append(float(te.get("loss", 0.0)))

    merged = []
    for ws, we in word_spans:
        b = buckets[(ws, we)]
        scores = b["scores"]
        
        text_slice = raw_text[ws:we]
        if not text_slice.strip(): 
            continue

        if not scores:
            score = float('nan')
            l_val = float('nan')
        else:
            if agg == "sum":
                score = float(sum(scores))
                l_val = float(sum(b["loss"]))
            elif agg == "mean":
                score = float(sum(scores) / len(scores))
                l_val = float(sum(b["loss"]) / len(b["loss"]))
            elif agg == "max":
                score = float(max(scores))
                l_val = float(max(b["loss"]))
            else:
                raise ValueError(f"Unknown agg={agg}")

        merged.append({
            "span": [ws, we],
            "text": text_slice,
            "score": score,
            "loss": l_val,
        })

    merged.sort(key=lambda x: (x["span"][0], x["span"][1]))
    return merged
