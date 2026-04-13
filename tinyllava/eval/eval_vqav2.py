import argparse
import json
import os

from tinyllava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-file", type=str, required=True)
    return parser.parse_args()


def load_annotations(annotation_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = {}
    for ann in data["annotations"]:
        gt_answers = [x["answer"] for x in ann["answers"]]
        annotations[ann["question_id"]] = gt_answers
    return annotations


def load_results(result_file):
    results = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def main():
    args = get_args()
    annotations = load_annotations(args.annotation_file)
    results = load_results(args.result_file)

    pred_list = []
    missing = 0
    for result in results:
        question_id = result["question_id"]
        if question_id not in annotations:
            missing += 1
            continue
        pred_list.append(
            {
                "pred_answer": result["text"],
                "gt_answers": annotations[question_id],
            }
        )

    evaluator = TextVQAAccuracyEvaluator()
    accuracy = evaluator.eval_pred_list(pred_list) if pred_list else 0.0

    print(os.path.splitext(os.path.basename(args.result_file))[0])
    print(f"Samples: {len(pred_list)}")
    print(f"Missing annotations: {missing}")
    print(f"Accuracy: {100.0 * accuracy:.2f}%")


if __name__ == "__main__":
    main()
