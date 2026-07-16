import argparse
import json
import hashlib
from collections import defaultdict
from copy import deepcopy
import numpy as np


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
from utils.ensemble_utils import *
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputs', nargs='+', required=True, help='input jsons from multiple models')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--agg', default='mean', choices=['mean', 'max', 'min'])
    ap.add_argument('--level', default='word', choices=['auto', 'word', 'sample'])
    ap.add_argument(
        '--no_normalize',
        action='store_true',
        help='Disable score normalization before ensemble aggregation.'
    )
    ap.add_argument('--top_k', type=int, default=None)
    ap.add_argument('--filter_ratio', type=float, default=None)

    args = ap.parse_args()

    first_obj = load_json(args.inputs[0])
    inferred = infer_level(first_obj)
    level = inferred if args.level == 'auto' else args.level

    if level == 'word':
        maps = []
        out_meta = None
        for p in args.inputs:
            spans, meta = load_word_items(p)
            if out_meta is None:
                out_meta = meta
            maps.append(
                make_word_map(
                    spans,
                    normalize=(not args.no_normalize)
                )
            )
        ens = aggregate_word_maps(maps, agg=args.agg)
        
        dump_word_results(
            ens,
            args.output,
            meta=out_meta,
            top_k=args.top_k,
            filter_ratio=args.filter_ratio
        )
    else:
        maps = []
        out_meta = None
        for p in args.inputs:
            items, meta = load_sample_items(p)
            if out_meta is None:
                out_meta = meta
            maps.append(make_sample_map(items, normalize=(not args.no_normalize)))
        ens = aggregate_sample_maps(maps, agg=args.agg)
        dump_sample_results(
            ens,
            args.output,
            meta=out_meta,
            top_k=args.top_k,
            filter_ratio=args.filter_ratio
        )


if __name__ == '__main__':
    main()
