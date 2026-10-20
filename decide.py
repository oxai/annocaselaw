"""Run a decision model on AnnoCaseLaw and save its probabilities to runs/.

    uv run decide.py --model typesafe-jev-1.13.0
    uv sync --group clef && uv run decide.py --model Cloudflare/clef

Each model answers one yes/no question per sentence and category, and per precedent
candidate, and returns the probability of "yes". Items with probability >= 0.5 are selected.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import httpx

import data

THRESHOLD = 0.5


def systemone(model, settings):
    """Jev, or Kev on its own server: both take a state and questions over HTTP."""
    key = os.environ[settings["api_key_env"]] if "api_key_env" in settings else "none"
    client = httpx.Client(base_url=settings["base_url"], timeout=600, headers={"Authorization": f"Bearer {key}"})

    def classify(state, questions):
        response = client.post("/systemone", json={"model": settings.get("served_as", model),
                                                   "state": state, "questions": questions})
        response.raise_for_status()
        return response.json()
    return classify


def clef(model, settings):
    """Clef and Clef Flash, run locally with the inference code published in their repositories.

    The paper's Clef outputs used an equivalent Triton kernel for one convolution, needed on our
    hardware, so a rerun may differ for items with probabilities very close to 0.5.
    """
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, revision=settings["revision"])
    sys.path.insert(0, path)
    import joint_schema_model
    network, processor = joint_schema_model.load_release_model(path)
    limit = json.loads((Path(path) / "config.json").read_text())["text_config"]["max_position_embeddings"]

    def classify(state, questions):
        # Encode without the default length cap, which would silently cut long opinions.
        response = joint_schema_model.systemone(network, processor, {"model": model, "state": state,
                                                                     "questions": questions}, max_length=10**9)
        if response["usage"]["input_tokens"] > limit:
            raise ValueError(f"input of {response['usage']['input_tokens']} tokens exceeds the context of {limit}")
        return response
    return classify


def probabilities(response, questions):
    if response.get("truncated") or set(response["answers"]) != set(questions):
        raise ValueError("the response is truncated or does not answer every question")
    return {qid: answer["noul"] for qid, answer in response["answers"].items()}


def finish(opinion, answers):
    """Probabilities from all requests -> one result line, selecting at the threshold."""
    merged = {qid: p for answer in answers for qid, p in answer.items()}
    sentence = [{category: merged[f"{i}.{key}"] for key, category in data.QUESTION_KEYS.items()}
                for i in range(1, len(opinion["sentences"]) + 1)]
    precedent = [merged[str(i)] for i in range(1, len(opinion["precedent_candidates"]) + 1)]
    selected = {category: [i for i, p in enumerate(sentence, 1) if p[category] >= THRESHOLD]
                for category in data.CATEGORIES}
    relevant = [i for i, p in enumerate(precedent, 1) if p >= THRESHOLD]
    return data.to_result(opinion, selected, relevant) | {"sentence_probabilities": sentence,
                                                         "precedent_probabilities": precedent}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="identifier from models.yaml")
    parser.add_argument("--opinions-from", type=Path, help="run only the opinions in this result file")
    parser.add_argument("--workers", type=int, default=8, help="concurrent requests (Clef runs one at a time)")
    args = parser.parse_args()

    settings = data.models()[args.model]
    classify = {"systemone": systemone, "clef": clef}[settings["backend"]](args.model, settings)
    opinions = data.opinions()
    if args.opinions_from:
        opinions = {cid: opinions[cid] for cid in data.read_results(args.opinions_from)}

    def call(state, questions):
        return probabilities(classify(state, questions), questions)

    workers = 1 if settings["backend"] == "clef" else args.workers
    data.run(opinions, data.decision_requests, call, finish, data.result_path(args.model, folder=data.RUNS), workers)


if __name__ == "__main__":
    main()
