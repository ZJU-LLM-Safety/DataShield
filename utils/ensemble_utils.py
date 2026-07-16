import argparse
import json
import hashlib
from collections import defaultdict
from copy import deepcopy
import numpy as np


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def robust_scale_norm(scores, clip_q=(1, 99), eps=1e-12):
    """
    Normalize difference scores while preserving zero semantics, using clipped
    standard-deviation scaling to align magnitudes across models.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores

    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return np.full_like(scores, np.nan, dtype=np.float64)

    finite_scores = scores[finite_mask]
    
    # Clip outliers before computing a robust standard deviation.
    lo, hi = np.nanpercentile(finite_scores, clip_q)
    clipped = np.clip(finite_scores, lo, hi)
    
    # Compute the scale without centering, so zero remains unchanged.
    std_val = np.std(clipped)
    if std_val < eps:
        std_val = 1.0

    out = np.full_like(scores, np.nan, dtype=np.float64)
    out[finite_mask] = clipped / std_val
    return out


def infer_level(obj):
    if isinstance(obj, dict) and 'level' in obj and 'data' in obj:
        level = obj['level']
        if level in ('word', 'token'):
            return 'word'
        if level == 'sample':
            return 'sample'

    if isinstance(obj, dict):
        if not obj:
            return 'word'
        vals = list(obj.values())
        if vals and isinstance(vals[0], list):
            return 'word'

    if isinstance(obj, list):
        return 'sample'

    raise ValueError('Unable to infer level from input JSON structure.')


def unwrap_payload(obj):
    if isinstance(obj, dict) and 'data' in obj and 'level' in obj:
        meta = {k: deepcopy(v) for k, v in obj.items() if k != 'data'}
        return obj['data'], meta
    return obj, {}


def stable_content_key(item):
    content = {
        k: item.get(k)
        for k in ['instruction', 'input', 'output', 'text']
        if k in item
    }
    if not content:
        content = {k: v for k, v in item.items() if k not in {'sim_score', 'score', 'n_models'}}
    blob = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return ('content_md5', hashlib.md5(blob.encode('utf-8')).hexdigest())


def stable_sample_key(item):
    if 'id' in item:
        return ('id', str(item['id']))
    if 'sample_idx' in item:
        return ('sample_idx', str(item['sample_idx']))
    return stable_content_key(item)


def coerce_sample_id(raw_key, items):
    if items and isinstance(items[0], dict):
        for field in ('id', 'sample_idx'):
            if field in items[0] and items[0].get(field) is not None:
                return str(items[0][field])
    return str(raw_key)


def load_word_items(path):
    obj = load_json(path)
    level = infer_level(obj)
    if level != 'word':
        raise ValueError(f'{path} is not a word/token-level JSON')
    data, meta = unwrap_payload(obj)

    spans = []
    dropped_non_finite = 0
    for raw_k, items in data.items():
        sample_key = coerce_sample_id(raw_k, items)
        for it in items:
            s, e = it['span']
            score = float(it['score'])
            if not np.isfinite(score):
                dropped_non_finite += 1
                continue
            spans.append({
                'sample_id': sample_key,
                'span': [int(s), int(e)],
                'text': it.get('text', ''),
                'score': score,
            })

    if dropped_non_finite > 0:
        print(f'[Warn] {path}: dropped {dropped_non_finite} non-finite word/token scores')
    return spans, meta


def make_word_map(spans, normalize=True):
    if not spans:
        return {}

    spans = deepcopy(spans)

    if normalize:
        valid_scores = [float(x['score']) for x in spans if np.isfinite(float(x['score']))]
        if valid_scores:
            # Use zero-preserving robust scaling instead of robust_minmax_norm.
            norm = robust_scale_norm(valid_scores, clip_q=(1, 99))
            idx = 0
            for x in spans:
                if np.isfinite(float(x['score'])):
                    x['score'] = float(norm[idx])
                    idx += 1

    mp = {}
    for x in spans:
        if not np.isfinite(float(x['score'])):
            continue
        
        sid = x['sample_id']
        span_tuple = tuple(x.get('span', [0, 0]))
        
        key = (sid, span_tuple)
        mp[key] = {
            'text': x.get('text', ''),
            'score': float(x['score']),
            'sample_id': sid,
            'span': list(span_tuple),
        }
    return mp


def aggregate_word_maps(maps, agg='mean'):
    all_spans = defaultdict(list)
    
    for m_idx, mp in enumerate(maps):
        for k, v in mp.items():
            sid = v['sample_id']
            if not np.isfinite(v['score']): continue
            v_copy = deepcopy(v)
            v_copy['model_idx'] = m_idx
            all_spans[sid].append(v_copy)

    grouped_by_sample = defaultdict(list)

    for sid, spans in all_spans.items():
        spans.sort(key=lambda x: x['span'][0])
        merged_groups = []
        
        for sp in spans:
            s, e = sp['span']
            placed = False
            for group in merged_groups:
                gs, ge = group['span']
                if max(s, gs) < min(e, ge):
                    group['spans'].append(sp)
                    group['span'] = [min(s, gs), max(e, ge)]
                    placed = True
                    break
            
            if not placed:
                merged_groups.append({
                    'span': [s, e],
                    'spans': [sp]
                })
                
        for group in merged_groups:
            model_scores = defaultdict(list)
            for sp in group['spans']:
                model_scores[sp['model_idx']].append(sp['score'])
            
            avg_model_scores = [sum(lst)/len(lst) for lst in model_scores.values()]
            
            if not avg_model_scores:
                continue
                
            if agg == 'mean': s = sum(avg_model_scores) / len(avg_model_scores)
            elif agg == 'max': s = max(avg_model_scores)
            elif agg == 'min': s = min(avg_model_scores)
            
            best_text = group['spans'][0].get('text', '')
            for sp in group['spans']:
                if len(sp.get('text', '')) > len(best_text):
                    best_text = sp.get('text', '')

            grouped_by_sample[sid].append({
                'sample_id': sid,
                'span': group['span'],
                'text': best_text,
                'absolute_score': float(s),
                'n_models': len(model_scores)
            })

    out = []
    for sid, items in grouped_by_sample.items():
        items.sort(key=lambda x: x['span'][0])
        for i, item in enumerate(items):
            out.append({
                'sample_id': sid,
                'span': item['span'],
                'text': item['text'],
                'score': float(item['absolute_score']),
                'n_models': item['n_models'],
                'word_idx': i
            })

    out.sort(key=lambda x: (str(x['sample_id']), x.get('word_idx', 0)))
    return out


def load_sample_items(path):
    obj = load_json(path)
    level = infer_level(obj)
    if level != 'sample':
        raise ValueError(f'{path} is not a sample-level JSON')
    data, meta = unwrap_payload(obj)
    if not isinstance(data, list):
        raise ValueError(f'{path}: sample-level data should be a list')

    items = []
    dropped_non_finite = 0
    for row in data:
        item = deepcopy(row)
        if 'sim_score' not in item:
            raise ValueError(f'{path}: sample-level item missing sim_score')
        item['sim_score'] = float(item['sim_score'])
        if not np.isfinite(item['sim_score']):
            dropped_non_finite += 1
            continue
        items.append(item)

    if dropped_non_finite > 0:
        print(f'[Warn] {path}: dropped {dropped_non_finite} non-finite sample scores')
    return items, meta


def make_sample_map(items, normalize=True):
    if not items:
        return {}
    items = deepcopy(items)
    if normalize:
        # Use zero-preserving robust scaling instead of robust_minmax_norm.
        norm = robust_scale_norm([x['sim_score'] for x in items], clip_q=(1, 99))
        for x, s in zip(items, norm):
            x['sim_score'] = float(s)

    mp = {}
    for item in items:
        if not np.isfinite(item['sim_score']):
            continue
        key = stable_sample_key(item)
        payload = deepcopy(item)
        mp[key] = payload
    return mp


def aggregate_sample_maps(maps, agg='mean'):
    acc = defaultdict(list)
    ref = {}
    for mp in maps:
        for k, v in mp.items():
            s = float(v['sim_score'])
            if not np.isfinite(s):
                continue
            acc[k].append(s)
            if k not in ref:
                ref[k] = deepcopy(v)

    out = []
    for k, lst in acc.items():
        lst = [float(v) for v in lst if np.isfinite(v)]
        if not lst:
            continue

        if agg == 'mean':
            s = float(sum(lst) / len(lst))
        elif agg == 'max':
            s = float(max(lst))
        elif agg == 'min':
            s = float(min(lst))
        else:
            raise ValueError('agg must be one of: mean|max|min')

        item = deepcopy(ref[k])
        item['sim_score'] = s
        item['n_models'] = len(lst)
        out.append(item)
    return out


def dump_word_results(spans, out_path, meta=None, top_k=None, filter_ratio=None):
    spans = [x for x in spans if np.isfinite(float(x['score']))]
    if top_k is not None or filter_ratio is not None:
        spans.sort(key=lambda x: x['score'], reverse=True)

    if top_k is not None:
        spans = spans[:max(0, int(top_k))]
    elif filter_ratio is not None:
        cutoff = int(len(spans) * float(filter_ratio))
        cutoff = max(1, min(len(spans), cutoff)) if len(spans) > 0 else 0
        spans = spans[:cutoff]

    results_by_sample = defaultdict(list)
    for x in spans:
        sample_key = str(x['sample_id'])
        entry = {
            'span': x['span'],
            'text': x['text'],
            'score': x['score'],
            'n_models': x.get('n_models'),
        }
        if 'word_idx' in x:
            entry['word_idx'] = x['word_idx']
        results_by_sample[sample_key].append(entry)

    for k in results_by_sample:
        results_by_sample[k].sort(key=lambda z: z.get('word_idx', 10**18))

    if meta:
        payload = deepcopy(meta)
        payload['data'] = dict(results_by_sample)
    else:
        payload = dict(results_by_sample)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def dump_sample_results(items, out_path, meta=None, top_k=None, filter_ratio=None):
    items = [x for x in items if np.isfinite(float(x['sim_score']))]
    if top_k is not None or filter_ratio is not None:
        items.sort(key=lambda x: x['sim_score'], reverse=True)

    if top_k is not None:
        items = items[:max(0, int(top_k))]
    elif filter_ratio is not None:
        cutoff = int(len(items) * float(filter_ratio))
        cutoff = max(1, min(len(items), cutoff)) if len(items) > 0 else 0
        items = items[:cutoff]

    if meta:
        payload = deepcopy(meta)
        payload['data'] = items
    else:
        payload = items

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

