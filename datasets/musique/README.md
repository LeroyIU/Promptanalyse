# Dataset Card for MuSiQue

## Dataset Summary
MuSiQue (*Multi-hop Questions via Single-hop Question Composition*) is a multi-hop
open-domain QA dataset of ~25k questions, each composed from 2–4 single-hop
questions drawn from SQuAD, T-REx, Natural Questions, MLQA and Zero Shot RE.
Every question ships with **20 Wikipedia paragraphs**: the 2–4 paragraphs that
carry the reasoning chain (`is_supporting: true`) plus 16–18 topically related
distractors. Questions are filtered against reasoning shortcuts, so a model
cannot skip a hop — every supporting paragraph is genuinely required.

Two variants are released: **MuSiQue-Ans** (every question answerable from its
paragraphs) and **MuSiQue-Full**, which pairs each answerable question with a
minimally-changed unanswerable twin.

## Why this dataset in this project
PopQA is *closed-book*: the prompt is question-only, so there is little to
compress and no ground truth about what compression removed. MuSiQue supplies
the with-context condition:

- **Long, realistic prompts.** 20 paragraphs are ~2.5–3k tokens per prompt
  (~11k words in the LongBench packaging), which is the regime prompt
  compression is actually used in.
- **Gold-evidence labels.** `is_supporting` per paragraph and
  `paragraph_support_idx` per hop mean a compressed prompt can be scored on
  *what it kept*, not just on whether the final answer was still correct.
- **Built-in redundancy.** The 16–18 distractors are naturally occurring,
  non-synthetic redundancy — a reference point next to the redundancy the
  Redundanzgenerator injects deliberately.
- **Shortcut-free by construction.** Losing one supporting paragraph makes the
  answer underivable, so compression damage shows up in accuracy instead of
  being masked by a lucky guess.
- **Established baseline.** MuSiQue is one of the multi-document QA tasks in
  LongBench and is used throughout the prompt-compression literature
  (LongLLMLingua, LLMLingua-2), so results are comparable.

## Languages
English only.

## Dataset Structure
### Data Instances
- MuSiQue-Ans: ~25k questions (19,938 train, ~2.4k dev, ~2.4k test)
- MuSiQue-Full: roughly double that — each answerable question is paired with
  an unanswerable twin
- 20 paragraphs per question, 2–4 of them supporting

The `test` splits are released without answers or supporting labels (leaderboard
evaluation) and are therefore not usable here — **use the `dev` split**.

## Data Fields
- `id`: question id; the prefix encodes the composition shape (`2hop__`,
  `3hop1__`, `4hop3__`, …)
- `question`: the composed multi-hop question
- `question_decomposition`: list of single-hop steps, each with `id`,
  `question`, `answer` and `paragraph_support_idx` (index into `paragraphs`)
- `paragraphs`: list of 20 passages, each with `idx`, `title`,
  `paragraph_text` and `is_supporting`
- `answer`: gold answer string
- `answer_aliases`: alternative surface forms of the gold answer
- `answerable`: whether the question is answerable from its paragraphs
  (always `true` in MuSiQue-Ans)

## Download
The data is not committed to this repository. Fetch it into this directory:

```bash
python datasets/musique/fetch_musique.py            # official release via gdown
python datasets/musique/fetch_musique.py --via hf   # Hugging Face mirror
```

This writes `musique_ans_v1.0_dev.jsonl` (and `musique_full_v1.0_dev.jsonl`)
here. The `train`/`test` files are gitignored — they are large and, for `test`,
unlabelled.

## Stand der Integration
Fertig und getestet:
- `MuSiQueLoader` im Redundanzgenerator (Indizierung nach Hop-Zahl, Gold-/
  Distraktor-Trennung, Antworten inkl. Aliase, Prompt-Bau)
- `ContextPassage` in `FewShotPrompt`/`Demonstration` plus Rendering als
  nummerierter `Context:`-Block
- CLI: `redundanzgen build-context`

Noch offen (bewusst nicht mitentschieden):
- **Pipeline-Anbindung.** `PipelineStore.run_pipeline` baut Roh-Prompts bislang
  fest aus PopQA; für einen MuSiQue-Lauf braucht es einen zweiten
  `build_raw_prompt`-Pfad.
- **Lexikalische Redundanz.** Für MuSiQue gibt es kein Paraphrasen-Pendant zu
  PopQA-TP. Optionen: Query-Paraphrasen per LLM erzeugen (wie bereits bei den
  Instruktionen), oder die lexikalische Bedingung im Kontext-Setting durch
  Passagen-Redundanz ersetzen (Duplikate/Umformulierungen der Supporting-
  Passagen) — das wäre allerdings eine Änderung am Faktordesign und keine
  reine Implementierungsfrage.

## Licensing Information
MuSiQue is distributed under a [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

**Usage caution (from the authors):** MuSiQue was built by composing questions
from seed single-hop datasets (SQuAD, T-REx, Natural Questions, MLQA, Zero Shot
RE). Single-hop questions used in MuSiQue's dev/test sets may appear in those
datasets' training sets. If a seed dataset is used in any way (e.g. for
pretraining), exclude the ids listed in `dev_test_singlehop_questions_v1.0.json`.

## Citation Information
```
@article{trivedi2022musique,
  title={MuSiQue: Multihop Questions via Single-hop Question Composition},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal={Transactions of the Association for Computational Linguistics},
  volume={10},
  pages={539--554},
  year={2022}
}
```
