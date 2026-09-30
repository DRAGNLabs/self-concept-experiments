import json


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
