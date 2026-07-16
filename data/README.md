# Data

This directory contains the released DataShield data files used by the pipeline examples.

## Files

| Path | Format | Rows | Purpose |
| :--- | :--- | ---: | :--- |
| `anchors/pure-bad-100.jsonl` | JSONL | 99 | Unsafe anchor messages for building the unsafe reference subspace. |
| `anchors/pure-bad-100-anchor1.jsonl` | JSONL | 99 | Safe/refusal anchor messages for building the safe reference subspace. |
| `train_data/alpaca-gpt4_no_safety.json` | JSON array | 45,774 | Instruction-tuning data for scanning and fine-tuning examples. |
| `train_data/dolly_no_safety.json` | JSON array | 14,624 | Instruction-tuning data for scanning and fine-tuning examples. |

## Training Data Schema

`train_data/*.json` files are JSON arrays. Each item should contain:

```json
{
  "instruction": "Task instruction",
  "input": "Optional task input",
  "output": "Assistant response"
}
```

Some files may include extra metadata fields such as `category` or a preformatted `text` field. The pipeline reads `instruction`, `input`, and `output`.

## Anchor Schema

`anchors/*.jsonl` files contain one JSON object per line:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

The projection stage uses the `messages` field with the selected reference model's chat template.

## Generated Mask Schema

Word-level projection and ensemble files use a dictionary keyed by sample id:

```json
{
  "level": "word",
  "data": {
    "0": [
      {
        "span": [0, 8],
        "text": "example",
        "score": 1.23,
        "n_models": 2
      }
    ]
  }
}
```

`span` values are character offsets relative to the response text. During fine-tuning, DataShield resolves these character spans against the target model tokenizer and masks the corresponding response-token losses.

## Safety Note

These files are provided for controlled research and reproducibility. Some records are safety-sensitive because they are used to build or evaluate safety-critical subspaces. Avoid redistributing modified versions without the same safety context, and do not use the data to train systems for harmful behavior.
