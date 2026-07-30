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

## Injected redundancy for compression experiments
`inject_redundancy.py` takes the 8k+ subset above and produces five JSONL
variants for controlled prompt-compression experiments:

- `triviaqa_baseline.jsonl` — unmodified passthrough of the 100 samples
- `triviaqa_redundancy_lexical.jsonl` — paraphrases of selected passage
  sentences inserted right after the original sentence
- `triviaqa_redundancy_demonstration.jsonl` — whole few-shot demonstrations
  duplicated (verbatim, or paraphrased with `--near-duplicate`) and
  reinserted at a randomized, non-adjacent position
- `triviaqa_redundancy_instruction.jsonl` — the task instruction (LongBench's
  own `triviaqa` template from `dataset2prompt.json`, which appears nowhere
  in `context`/`input` themselves) repeated at several positions, verbatim
  and/or paraphrased
- `triviaqa_redundancy_combined.jsonl` — all three stacked

Every non-baseline row adds `redundancy_spans` (per category: exact
character span in the new `context` that was injected, plus the source text
it duplicates/paraphrases — `{start, end, type, original_text,
inserted_text}`), `original_length`/`augmented_length` (tiktoken
`cl100k_base` token counts) and `redundancy_ratio`, so a compressor's output
can later be checked against exactly what should be removable without
losing the answer.

```bash
python datasets/longbench_trivia_qa_e/inject_redundancy.py
```

Paraphrasing (lexical redundancy, the combined variant, `--near-duplicate`
demonstrations, and paraphrased instruction copies) calls the Anthropic API;
set `ANTHROPIC_API_KEY` in the environment (never hardcoded — see
`AnthropicParaphraser` in the script). Variants that don't need paraphrasing
(verbatim demonstration duplication, verbatim instruction repetition) run
without a key. When no key is configured the script prints a warning,
downgrades `--instruction-mode` to `verbatim`, and skips the `lexical` and
`combined` variants rather than failing outright.

Configurable via CLI: `--lexical-ratio` (0.15), `--include-answer-sentences`,
`--demo-duplicate-ratio` (0.2), `--near-duplicate`, `--instruction-repeats`
(2), `--instruction-mode` (`verbatim`/`paraphrased`/`both`, default `both`),
`--seed` (42), `--conditions` (subset of `lexical demonstration instruction
combined` to regenerate), `--paraphrase-model`.

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
