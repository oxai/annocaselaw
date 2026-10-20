# AnnoCaseLaw

Code for running and scoring models on AnnoCaseLaw, a densely annotated benchmark of 471 U.S.
civil-negligence court opinions with more than 12,000 expert highlights across five annotation
categories. Dataset: [riltonfranzone/annocaselaw](https://huggingface.co/datasets/riltonfranzone/annocaselaw).

## Score

```bash
uv sync
uv run score.py                              # predictions reported in the paper (results/)
uv run score.py runs/<model>.jsonl           # your own run
```

`results/` holds the predictions behind the paper's tables, so scores can be checked without
rerunning models. Add `--opinions-from results/gemma-4-31B-it.yes-no.jsonl` to score on the
40 opinions of the appendix comparison.

## Run a model

```bash
export DO_MODEL_ACCESS_KEY=...                   # DigitalOcean inference
uv run generate.py --model openai-gpt-5.6-sol    # generative models
uv run decide.py --model typesafe-jev-1.13.0     # decision models
```

Model identifiers and settings are in `models.yaml`, including how to serve the open-weight
models. Runs are saved to `runs/` and resume if interrupted. Reruns will not match `results/`
exactly: API models are not versioned and several models sample.

## Results format

One file per model, one line per opinion; sentences and candidates follow the dataset's order.

```json
{"id": "0001", "sentence_labels": [[], ["Procedural History"], ["Outcome"], ["Facts"], ...],
 "precedent_relevant": [false, false, false, true, ...]}
```

Decision models also store `sentence_probabilities` and `precedent_probabilities`, and select
items with probability at least 0.5.

## Licence and citation

MIT. Opinion text is from the Caselaw Access Project.

```bibtex
@inproceedings{annocaselaw2026, ...}
```
