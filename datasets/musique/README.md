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
MuSiQue is the sole data basis of the experiment. It replaced PopQA, which is
*closed-book*: a question-only prompt leaves little to compress and yields no
ground truth about what compression removed.

- **Long, realistic prompts.** Gemessen an den fertig gerenderten Prompts
  dieses Repos: Median 1.681 Whitespace-Tokens (Spanne 1.059–2.079). In der
  LongBench-Verpackung, die zusätzlich Distraktoren aus anderen Fragen
  hinzuzieht, sind es ~11k Wörter. Beides ist das Regime, in dem
  Prompt-Kompression tatsächlich eingesetzt wird.
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
- MuSiQue-Ans: ~25k questions (19,938 train, 2,417 dev, ~2.4k test)
- MuSiQue-Full: roughly double that — each answerable question is paired with
  an unanswerable twin
- 20 paragraphs per question, 2–4 of them supporting

Verified against the committed dev split: 2,417 records, hop distribution
2/3/4 = 1252/760/405, the number of supporting paragraphs equals the hop count
in every record, and 2,401 of 2,417 carry the full 20 paragraphs (the rest 17–19).

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
`musique_ans_v1.0_dev.jsonl` (30 MB) and `musique_ans_v1.0_train.jsonl` (241 MB)
are committed to this repository via **Git LFS**. After cloning:

```bash
git lfs install
git lfs pull
```

Without Git LFS the files contain only pointer text (~130 bytes) instead of the
data, and the loader will fail on the first line.

To fetch the data from its source instead — e.g. to get MuSiQue-Full or the
`dev_test_singlehop_questions_v1.0.json` id list:

```bash
python datasets/musique/fetch_musique.py            # official release via gdown
python datasets/musique/fetch_musique.py --via hf   # Hugging Face mirror
```

The `test` split is gitignored: it ships without answers or supporting labels
(leaderboard evaluation) and is unusable here.

## Stand der Integration
Die Pipeline läuft vollständig auf MuSiQue:
- `MuSiQueLoader` im Redundanzgenerator (Indizierung nach Hop-Zahl, Gold-/
  Distraktor-Trennung, Antworten inkl. Aliase, Prompt-Bau)
- `ContextPassage` in `FewShotPrompt`/`Demonstration` plus Rendering als
  nummerierter `Context:`-Block
- Redundanztypen `passages` / `demonstrations` / `instructions` — je einer pro
  Bestandteil eines Kontext-Prompts
- `PipelineStore.run_pipeline` baut Roh-Prompts aus MuSiQue und schreibt neben
  `is_correct` die Evidenz-Metriken `supporting_retained`,
  `distractor_retained` und `evidence_selectivity` ins Manifest
- CLI: `redundanzgen build` / `redundanzgen generate`

Zum Faktordesign: Die frühere Bedingung `lexical` (Query-Paraphrasen aus
PopQA-TP) ist durch `passages` ersetzt. MuSiQue hat kein Paraphrasen-Pendant zu
PopQA-TP, und im Kontext-Setting ist die Passagenebene ohnehin der Ort, an dem
Redundanz für die Kompression eine Rolle spielt: Der Kontext ist der weitaus
größte Teil des Prompts. `passage_mode="restate"` behält dabei den lexikalischen
Charakter (dieselbe Information, andere Formulierung), `"duplicate"` ist der
wörtliche Grenzfall.

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
