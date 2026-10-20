"""Score saved predictions against the AnnoCaseLaw reference and print Table 1.

    uv run score.py                                   # Table 1: every model in results/
    uv run score.py --opinions-from results/gemma-4-31B-it.yes-no.jsonl   # appendix control, 40 opinions
"""
import argparse
from pathlib import Path

import numpy as np

import data

COLUMNS = {"ALF": "Application of Law to Facts", "PH": "Procedural History", "Precedents": "Relevant Precedents",
           "Facts": "Facts", "Outcome": "Outcome"}
RESAMPLES, SEED = 10000, 0


def reference(opinion, category):
    """Reference items and the IDs excluded from scoring."""
    if category == "Relevant Precedents":
        return {i for i, c in enumerate(opinion["precedent_candidates"], 1) if c["relevant"]}, set()
    labels = [s["labels"] for s in opinion["sentences"]]
    return ({i for i, l in enumerate(labels, 1) if l and category in l},
            {i for i, l in enumerate(labels, 1) if l is None})


def selected(result, category):
    if category == "Relevant Precedents":
        return {i for i, flag in enumerate(result["precedent_relevant"], 1) if flag}
    return {i for i, labels in enumerate(result["sentence_labels"], 1) if category in labels}


def f1_per_opinion(opinions, results, category):
    """F1 for each opinion with at least one reference item, keyed by opinion ID."""
    scores = {}
    for cid, opinion in opinions.items():
        gold, excluded = reference(opinion, category)
        if gold:
            predicted = selected(results[cid], category) - excluded
            tp = len(gold & predicted)
            scores[cid] = 2 * tp / (len(gold) + len(predicted))
    return scores


def draws(n, rng):
    """Bootstrap resamples of n opinions, drawn in blocks of 256."""
    return np.concatenate([rng.integers(0, n, size=(min(256, RESAMPLES - i), n))
                           for i in range(0, RESAMPLES, 256)])


def interval(values):
    low, high = np.quantile(values, [.025, .975])
    return (high - low) / 2


def score(opinions, results):
    """Per-category F1 and Macro-F1 (in %) with 95% bootstrap half-widths, resampling opinions."""
    row, per_category = {}, {}
    for column, category in COLUMNS.items():
        scores = f1_per_opinion(opinions, results, category)
        f1 = np.array(list(scores.values()))
        row[column] = (100 * f1.mean(), 100 * interval(f1[draws(len(f1), np.random.default_rng(SEED))].mean(axis=1)))
        per_category[category] = scores

    # Macro-F1: resample opinions once per draw for all categories, then average each
    # category over its eligible sampled opinions.
    ids = sorted(set().union(*per_category.values()))
    eligible = np.array([[cid in s for s in per_category.values()] for cid in ids])
    f1 = np.array([[s.get(cid, 0.) for s in per_category.values()] for cid in ids])
    rng, macro = np.random.default_rng(SEED), []
    while sum(map(len, macro)) < RESAMPLES:
        sample = rng.integers(0, len(ids), size=(min(256, RESAMPLES - sum(map(len, macro))), len(ids)))
        counts = eligible[sample].sum(axis=1)
        keep = (counts > 0).all(axis=1)
        macro.append((f1[sample[keep]].sum(axis=1) / counts[keep]).mean(axis=1))
    estimate = np.mean([np.mean(list(s.values())) for s in per_category.values()])
    row["Macro-F1"] = (100 * estimate, 100 * interval(np.concatenate(macro)))
    return row


def select_all(opinions):
    return {cid: {"sentence_labels": [list(data.CATEGORIES)] * len(o["sentences"]),
                  "precedent_relevant": [True] * len(o["precedent_candidates"])} for cid, o in opinions.items()}


def display_name(path, names):
    stem = Path(path).name.removesuffix(".jsonl")
    model, yes_no = stem.removesuffix(".yes-no").replace("--", "/"), stem.endswith(".yes-no")
    return names.get(model, model) + (" (yes/no)" if yes_no else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="*", type=Path, help="result files (default: all in results/)")
    parser.add_argument("--opinions-from", type=Path, help="score only the opinions in this result file")
    args = parser.parse_args()

    opinions = data.opinions()
    if args.opinions_from:
        opinions = {cid: opinions[cid] for cid in sorted(data.read_results(args.opinions_from))}
    names = {model: entry["name"] for model, entry in data.models().items()}
    order = {name: i for i, name in enumerate(names.values())}
    paths = args.results or sorted(data.RESULTS.glob("*.jsonl"))

    rows = []
    for path in paths:
        results = data.read_results(path)
        if opinions.keys() - results.keys():
            if args.results:
                raise SystemExit(f"{path} covers {len(results.keys() & opinions.keys())} of {len(opinions)} opinions")
            print(f"Skipping {path.name}: covers {len(results.keys() & opinions.keys())} of {len(opinions)} opinions")
            continue
        rows.append((display_name(path, names), score(opinions, results)))
    rows.sort(key=lambda r: order.get(r[0].removesuffix(" (yes/no)"), len(order)))
    rows.append(("Select all", score(opinions, select_all(opinions))))

    print(f"\nF1 (%) on {len(opinions)} opinions, ± half-width of the 95% bootstrap interval\n")
    width = max(len(name) for name, _ in rows)
    print(f"{'Model':<{width}}" + "".join(f"{column:>14}" for column in [*COLUMNS, "Macro-F1"]))
    for name, row in rows:
        print(f"{name:<{width}}" + "".join(f"{f'{v:.1f} ±{h:.1f}':>14}" for v, h in row.values()))


if __name__ == "__main__":
    main()
