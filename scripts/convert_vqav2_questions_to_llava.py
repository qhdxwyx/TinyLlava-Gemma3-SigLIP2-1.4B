import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--image-prefix", type=str, required=True, help="e.g. COCO_val2014")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for item in questions:
            image_id = int(item["image_id"])
            image_name = f"{args.image_prefix}_{image_id:012d}.jpg"
            record = {
                "question_id": item["question_id"],
                "image": image_name,
                "text": item["question"] + "\nAnswer the question using a single word or phrase.",
                "category": "default",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
