import argparse
import base64
import json
import os
import random
import re
from pathlib import Path

import requests
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--sample-mode", type=str, choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_image_to_data_url(image_path):
    suffix = Path(image_path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def extract_json_block(text):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def parse_openai_like_response(resp):
    data = resp.json()
    if "error" in data:
        raise RuntimeError(json.dumps(data["error"], ensure_ascii=False))
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(json.dumps(data, ensure_ascii=False))
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        content = "\n".join(texts)
    if not isinstance(content, str):
        raise RuntimeError(json.dumps(data, ensure_ascii=False))
    return content


def build_judge_prompt(question, answer):
    return (
        "You are grading a visual question answering prediction.\n"
        "Given the image, the question, and the candidate answer, judge whether the candidate answer is correct.\n"
        "Treat semantically equivalent short answers as correct, but be strict about factual mistakes.\n"
        "Return only JSON with this schema:\n"
        '{"correct": 1 or 0, "reason": "short explanation"}\n\n'
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n"
    )


def judge_sample(base_url, api_key, judge_model, question, answer, image_path, temperature, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": judge_model,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_judge_prompt(question, answer)},
                    {"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}},
                ],
            }
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    content = parse_openai_like_response(resp)
    parsed = extract_json_block(content)
    parsed["correct"] = int(parsed.get("correct", 0))
    parsed["reason"] = str(parsed.get("reason", "")).strip()
    return parsed


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    questions = load_jsonl(args.question_file)
    question_map = {item["question_id"]: item for item in questions}
    results = load_jsonl(args.result_file)

    eligible = [row for row in results if row["question_id"] in question_map]
    if args.sample_mode == "random":
        rng = random.Random(args.seed)
        rng.shuffle(eligible)
    if args.max_samples > 0:
        eligible = eligible[: args.max_samples]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if output_path.exists():
        for row in load_jsonl(str(output_path)):
            done[row["question_id"]] = row

    pending = [row for row in eligible if row["question_id"] not in done]

    with open(output_path, "a", encoding="utf-8") as fout:
        for row in tqdm(pending, desc="LLM judge"):
            q = question_map[row["question_id"]]
            image_path = os.path.join(args.image_folder, q["image"])
            judged = judge_sample(
                args.base_url,
                api_key,
                args.judge_model,
                q["text"],
                row["text"],
                image_path,
                args.temperature,
                args.timeout,
            )
            record = {
                "question_id": row["question_id"],
                "image": q["image"],
                "question": q["text"],
                "pred_answer": row["text"],
                "correct": judged["correct"],
                "reason": judged["reason"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            done[row["question_id"]] = record

    judged_rows = [done[row["question_id"]] for row in eligible if row["question_id"] in done]
    accuracy = sum(row["correct"] for row in judged_rows) / len(judged_rows) if judged_rows else 0.0

    print(f"Judged samples: {len(judged_rows)}")
    print(f"Proxy accuracy: {accuracy * 100:.2f}%")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
