"""Load AnnoCaseLaw, build the model inputs used in the paper, and run models and save their results."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import time
from pathlib import Path

import yaml
from datasets import load_dataset

DATASET = "riltonfranzone/annocaselaw"
DATASET_REVISION = "8fd1d11ec2b6a0dc16a604304a5e3aa2c105a244"  # the release used in the paper
ROOT = Path(__file__).parent
RESULTS = ROOT / "results"  # predictions reported in the paper
RUNS = ROOT / "runs"  # new runs of generate.py and decide.py
CATEGORIES = ("Facts", "Procedural History", "Application of Law to Facts", "Outcome")
QUESTION_KEYS = {"facts": "Facts", "ph": "Procedural History",
                 "alf": "Application of Law to Facts", "outcome": "Outcome"}
MAX_QUESTIONS = 256  # per decision-model request
ATTEMPTS = 3


def opinions():
    """The 471 opinions, keyed by ID."""
    return {row["id"]: row for row in load_dataset(DATASET, split="test", revision=DATASET_REVISION)}


def models():
    return yaml.safe_load((ROOT / "models.yaml").read_text())


def template(name):
    return (ROOT / "prompts" / f"{name}.txt").read_text().rstrip("\n")


def sentence_texts(opinion):
    return [opinion["text"][s["start"]:s["end"]] for s in opinion["sentences"]]


# Generative models: one prompt per task.

def sentence_prompt(opinion):
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentence_texts(opinion), 1))
    return template("sentences").replace("{{NUMBERED_SENTENCES}}", numbered)


def precedent_prompt(opinion):
    numbered = "\n".join(f"[{i}] {c['name']} — " + "; ".join(c["citations"])
                         for i, c in enumerate(opinion["precedent_candidates"], 1))
    return (template("precedents").replace("{{CLEANED_OPINION}}", opinion["text"])
            .replace("{{NUMBERED_CANDIDATES}}", numbered))


# Decision models: the same instructions without the OUTPUT section, then one
# yes/no question per sentence and category, or per precedent candidate.

def paragraphs(opinion):
    """Numbered sentences grouped at line breaks, with whitespace inside each sentence collapsed."""
    groups, previous = [], None
    text = opinion["text"]
    for i, (s, raw) in enumerate(zip(opinion["sentences"], sentence_texts(opinion)), 1):
        first = s["start"] + len(raw) - len(raw.lstrip())
        if previous is None or "\n" in text[previous:first]:
            groups.append([])
        groups[-1].append((i, " ".join(raw.split())))
        previous = s["end"] - (len(raw) - len(raw.rstrip()))
    return groups


def sentence_questions(paragraph):
    questions = {}
    for i, sentence in paragraph:
        words = sentence.split()
        opening = " ".join(words[:15]) + (" …" if len(words) > 15 else "")
        for key, category in QUESTION_KEYS.items():
            questions[f"{i}.{key}"] = {
                "type": "noul",
                "criteria": {"true": f"sentence [{i}] meets the {category} definition above, including its scope",
                             "false": f"sentence [{i}] does not meet the {category} definition above"},
                "instructions": f'Sentence [{i}] begins: "{opening}". Following the instructions above, '
                                f"should sentence [{i}] be selected as {category}?"}
    return questions


def precedent_questions(opinion):
    return {str(i): {
        "type": "noul",
        "instructions": f"Following the instructions above, should candidate [{i}] ({c['name']}) "
                        "be selected as a relevant precedent?",
        "criteria": {"true": f"candidate [{i}] meets the Relevant Precedents definition above, including its scope",
                     "false": f"candidate [{i}] does not meet the Relevant Precedents definition above"}}
        for i, c in enumerate(opinion["precedent_candidates"], 1)}


def batches(questions):
    items = list(questions.items())
    return [dict(items[i:i + MAX_QUESTIONS]) for i in range(0, len(items), MAX_QUESTIONS)]


def decision_requests(opinion):
    """(state, questions) pairs: precedent requests first, then one or more per paragraph."""
    state = re.sub(r"\nOUTPUT\n.*?\n(?=OPINION\n)", "\n", precedent_prompt(opinion), count=1, flags=re.S)
    requests = [(state, q) for q in batches(precedent_questions(opinion))]

    instructions = template("sentences").split("\nOUTPUT\n")[0].strip().replace(
        "Read the entire supplied opinion.", "Read all of the supplied opinion text.")
    groups = paragraphs(opinion)
    state = instructions + "\n\nOPINION\n\n" + "\n".join(f"[{i}] {s}" for group in groups for i, s in group)
    for group in groups:
        requests += [(state, q) for q in batches(sentence_questions(group))]
    return requests


def yes_no_prompt(state, question):
    """A decision-model question as a prompt for a generative model."""
    return (state + "\n\nCURRENT DECISION\n" + question["instructions"]
            + "\n\nYes: " + question["criteria"]["true"]
            + "\nNo: " + question["criteria"]["false"]
            + "\n\nReturn only Yes or No.")


# Results: one JSON line per opinion, mirroring the dataset.

def result_path(model, yes_no=False, folder=RESULTS):
    return folder / (model.replace("/", "--") + (".yes-no" if yes_no else "") + ".jsonl")


def read_results(path):
    if not Path(path).exists():
        return {}
    return {r["id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}


def to_result(opinion, sentence_ids, precedent_ids):
    """Selected IDs per category -> labels per sentence and a flag per candidate."""
    labels = [[] for _ in opinion["sentences"]]
    for category in CATEGORIES:
        for i in sentence_ids[category]:
            labels[i - 1].append(category)
    selected = set(precedent_ids)
    return {"id": opinion["id"], "sentence_labels": labels,
            "precedent_relevant": [i in selected for i in range(1, len(opinion["precedent_candidates"]) + 1)]}


def append_result(path, row):
    path.parent.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sort_results(path):
    if not path.exists():
        return
    rows = sorted(read_results(path).values(), key=lambda r: r["id"])
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# Running: as in the paper, a failed or invalid response is retried with the same prompt (up to
# three attempts) and the first valid answer is kept. Each opinion is saved once all its calls succeed.

def retry(call, *args):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return call(*args)
        except Exception as error:
            if attempt == ATTEMPTS:
                raise
            print(f"Attempt {attempt} failed ({error}); retrying")
            time.sleep(10 * attempt)


def run(opinions, jobs, call, finish, path, workers):
    """Run `call` on each of `jobs(opinion)` and save `finish(opinion, answers)` for every opinion.

    Opinions already in `path` are skipped, so rerunning resumes an interrupted or partly failed run.
    """
    done = read_results(path)
    todo = {cid: jobs(opinion) for cid, opinion in opinions.items() if cid not in done}
    answers = {cid: [None] * len(items) for cid, items in todo.items()}
    remaining = {cid: len(items) for cid, items in todo.items()}
    failed = set()
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(retry, call, *item): (cid, i) for cid, items in todo.items()
                   for i, item in enumerate(items)}
        for future in as_completed(futures):
            cid, i = futures[future]
            try:
                answers[cid][i] = future.result()
            except Exception as error:
                if cid not in failed:
                    print(f"Opinion {cid} failed: {error}")
                failed.add(cid)
            remaining[cid] -= 1
            if not remaining[cid] and cid not in failed:
                append_result(path, finish(opinions[cid], answers[cid]))
                print(f"Saved opinion {cid}")
    sort_results(path)
    print(f"{len(done) + len(todo) - len(failed)} of {len(opinions)} opinions in {path}")
    if failed:
        raise SystemExit(f"{len(failed)} opinions failed; rerun the command to retry them")
