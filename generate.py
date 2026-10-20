"""Run a generative model on AnnoCaseLaw and save its selections to runs/.

    uv run generate.py --model openai-gpt-5.6-sol
    uv run generate.py --model gemma-4-31B-it --yes-no --opinions-from results/gemma-4-31B-it.yes-no.jsonl

The model's settings come from models.yaml. Each opinion gets two requests, one per task;
with --yes-no the model instead answers the decision models' questions one at a time.
"""
import argparse
import json
import os
from pathlib import Path

import httpx

import data


class Model:
    def __init__(self, name, settings):
        self.name, self.settings = name, settings
        key = os.environ[settings["api_key_env"]] if "api_key_env" in settings else "none"
        self.client = httpx.Client(base_url=settings["base_url"], timeout=1200,
                                   headers={"Authorization": f"Bearer {key}"})

    def post(self, url, body):
        response = self.client.post(url, json=body)
        response.raise_for_status()
        return response.json()

    def complete(self, prompt, request=None, grammar=None):
        settings = self.settings
        body = {"model": self.name, "messages": [{"role": "user", "content": prompt}], **(request or settings["request"])}
        if settings.get("endpoint") == "messages":
            response = self.post("/messages", body | ({"system": settings["system"]} if "system" in settings else {}))
            return "".join(block["text"] for block in response["content"] if block["type"] == "text")
        if "system" in settings:
            body["messages"].insert(0, {"role": "system", "content": settings["system"]})
        if grammar:
            body["structured_outputs"] = {"grammar": grammar}
        if "context" in settings:  # vLLM: keep input plus output within the context window
            tokens = self.post(settings["base_url"].removesuffix("/v1") + "/tokenize",
                               {key: body[key] for key in ("model", "messages", "chat_template_kwargs") if key in body}
                               | {"add_generation_prompt": True})["count"]
            body["max_tokens"] = min(body["max_tokens"], settings["context"] - tokens)
        return self.post("/chat/completions", body)["choices"][0]["message"]["content"]


def parse(text, keys, n_items):
    """The final JSON object in a response, as sorted IDs per key."""
    text = text.rsplit("</think>", 1)[-1]  # drop reasoning from models that think inline
    try:
        value, _ = json.JSONDecoder().raw_decode(text[text.rindex("{"):])
        value = {key.strip().casefold(): ids for key, ids in value.items()}
    except (ValueError, AttributeError):
        raise ValueError("the response does not end with a JSON object") from None
    if set(value) != {key.casefold() for key in keys}:
        raise ValueError(f"expected the keys {keys}")
    output = {}
    for key in keys:
        if not isinstance(value[key.casefold()], list):
            raise ValueError(f"{key}: expected a list of IDs")
        ids = [int(i) if isinstance(i, str) and i.isdigit() else i for i in value[key.casefold()]]
        if not all(type(i) is int and 1 <= i <= n_items for i in ids):
            raise ValueError(f"{key}: every ID must be an integer from 1 to {n_items}")
        output[key] = sorted(set(ids))
    return output


def grammar(n_items, keys):
    """JSON with the given keys, each an ascending list of distinct IDs from 1 to n_items."""
    parts = []
    for i, key in enumerate(keys):
        parts += [json.dumps("{" if i == 0 else ","), "ws", json.dumps(json.dumps(key)), "ws", '":"', "ws", "ids", "ws"]
    rules = ["root ::= " + " ".join(parts + [json.dumps("}")]),
             'ws ::= "" | " " | "\\n" | "\\n    " | "\\n        "']
    if not n_items:
        return "\n".join(rules + ['ids ::= "[]"'])
    rules.append('ids ::= "[" ws ("]" | item1)')
    for i in range(1, n_items):
        rules += [f'item{i} ::= "{i}" ws after{i} | item{i + 1}', f'after{i} ::= "]" | "," ws item{i + 1}']
    rules.append(f'item{n_items} ::= "{n_items}" ws "]"')
    return "\n".join(rules)


def selection_jobs(opinion):
    jobs = [(data.sentence_prompt(opinion), data.CATEGORIES, len(opinion["sentences"]))]
    if opinion["precedent_candidates"]:  # opinions without candidates need no request
        jobs.append((data.precedent_prompt(opinion), ("relevant",), len(opinion["precedent_candidates"])))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="identifier from models.yaml")
    parser.add_argument("--yes-no", action="store_true", help="answer the decision models' yes/no questions")
    parser.add_argument("--opinions-from", type=Path, help="run only the opinions in this result file")
    parser.add_argument("--workers", type=int, default=16, help="concurrent requests")
    args = parser.parse_args()

    model = Model(args.model, data.models()[args.model])
    opinions = data.opinions()
    if args.opinions_from:
        opinions = {cid: opinions[cid] for cid in data.read_results(args.opinions_from)}

    if args.yes_no:
        def jobs(opinion):
            return [(state, qid, question)
                    for state, questions in data.decision_requests(opinion) for qid, question in questions.items()]

        def call(state, qid, question):
            prompt = data.yes_no_prompt(state, question)
            answer = model.complete(prompt, model.settings["yes_no_request"]).strip().rstrip(".").casefold()
            if answer not in ("yes", "no"):
                raise ValueError(f"expected Yes or No, got {answer!r}")
            return qid, answer == "yes"

        def finish(opinion, answers):
            selected = [qid for qid, yes in answers if yes]
            sentence = {category: sorted(int(qid.split(".")[0]) for qid in selected
                                         if "." in qid and data.QUESTION_KEYS[qid.split(".")[1]] == category)
                        for category in data.CATEGORIES}
            return data.to_result(opinion, sentence, [int(qid) for qid in selected if "." not in qid])
    else:
        jobs = selection_jobs

        def call(prompt, keys, n_items):
            text = model.complete(prompt, grammar=grammar(n_items, keys) if model.settings.get("grammar") else None)
            return parse(text, keys, n_items)

        def finish(opinion, answers):
            return data.to_result(opinion, answers[0], answers[1]["relevant"] if len(answers) > 1 else [])

    data.run(opinions, jobs, call, finish, data.result_path(args.model, args.yes_no, data.RUNS), args.workers)


if __name__ == "__main__":
    main()
