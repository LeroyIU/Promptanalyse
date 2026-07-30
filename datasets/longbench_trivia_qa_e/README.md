# Dataset Card for LongBench-E TriviaQA (8k+ subset)

## Dataset Summary
[LongBench](https://github.com/THUDM/LongBench) is a long-context benchmark
covering 21 tasks across six categories (multi-doc QA, single-doc QA,
summarization, few-shot learning, synthetic tasks, code completion).
**LongBench-E** is a re-sampled variant of a subset of these tasks (marked
with an `_e` suffix) that draws roughly balanced groups of ~100 examples from
three context-length buckets — 0-4k, 4-8k and 8k+ words — so that model
performance can be compared *within* a length bucket instead of being
confounded by it.

This directory holds only the **8k+ bucket of the `triviaqa_e` task**: a
few-shot open-domain QA setup where the context is a retrieved passage plus
several few-shot demonstrations, long enough to push well past typical
"short-context" evaluation.

## Why this subset in this project
The 8k+ bucket is the length regime where prompt compression is actually
useful: short enough to still fit common context windows so a
compressed-vs-uncompressed comparison is meaningful, long enough that there
is real redundancy (few-shot demonstrations, retrieved passage boilerplate)
for a compressor to remove. `answers` gives exact-match/F1-style ground
truth, so compressed prompts can be scored on whether the answer survives.

## Source and extraction
`THUDM/LongBench` on the Hugging Face Hub ships as a single `data.zip` plus
a `LongBench.py` loading script — no Parquet auto-conversion is available for
it, and current `datasets` releases (>=4) refuse to execute loading scripts
at all (even with `trust_remote_code=True`, `load_dataset` raises
`RuntimeError: Dataset scripts are no longer supported`). `extract_triviaqa_8k.py`
therefore downloads `data.zip` via `huggingface_hub` and reads
`data/triviaqa_e.jsonl` out of it directly — the same file the loading
script would have read.

Note on naming: the actual LongBench-E config for TriviaQA is `triviaqa_e`
(no underscore in "triviaqa"), not `trivia_qa_e`.

Each row in the source file already carries a `length` field (word count of
the rendered context) that LongBench used to build the three buckets. The
three buckets are cleanly separated at the 4000/8000 thresholds (100
examples each, verified against the full 300-row `triviaqa_e` split), so
`length >= 8000` is used directly as the bucket filter instead of
re-tokenizing the context.

```bash
python datasets/longbench_trivia_qa_e/extract_triviaqa_8k.py
```

Regenerates `triviaqa_longbench_e_8k_plus.jsonl` in this directory (default
`--output`).

## Data Fields
Same fields as the upstream LongBench(-E) schema, minus the two that are
constant across this file (`dataset` is always `"triviaqa_e"`, `language` is
always `"en"`):
- `context`: the retrieved passage(s) the question is grounded in
- `input`: the few-shot-formatted prompt (demonstrations + passage + question)
- `answers`: list of acceptable gold answer strings/aliases
- `length`: LongBench's own word-count bucket key (all rows here are >= 8000)
- `all_classes`: `null` for TriviaQA (only populated for classification tasks
  like `trec_e`/`lsht_e`)
- `_id`: LongBench's row id, kept for dedup/traceability

## Verified stats (100 rows)
- `length` (LongBench word count): min 8017, max 15960, mean ≈ 11294
- `context` word count (`.split()`): min 7164, max 15345, mean ≈ 10733
- `context` token count (`tiktoken`, `cl100k_base`): min 10303, max 22124,
  mean ≈ 15455
- No duplicate `_id`, no empty `answers`

## Licensing Information
LongBench is released under the [MIT License](https://github.com/THUDM/LongBench/blob/main/LICENSE).
The underlying TriviaQA passages/questions retain their original licensing
(TriviaQA is released under a [CC BY-SA 3.0 / Apache 2.0 mix](https://nlp.cs.washington.edu/triviaqa/); see the original dataset for details).

## Citation Information
```
@misc{bai2023longbench,
  title={LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding},
  author={Yushi Bai and Xin Lv and Jiajie Zhang and Hongchang Lyu and Jiankai Tang and Zhidian Huang and Zhengxiao Du and Xiao Liu and Aohan Zeng and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
  year={2023},
  eprint={2308.14508},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
