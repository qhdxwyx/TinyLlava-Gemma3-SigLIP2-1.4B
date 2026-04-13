import argparse
from pathlib import Path

import torch

from tinyllava.eval.run_tiny_llava import eval_model


def default_model_path() -> str:
    factory_root = Path(__file__).resolve().parent
    default_path = (
        factory_root
        / "checkpoints"
        / "llava_factory"
        / "tiny-llava-gemma-3-1b-pt-siglip2-so400m-patch14-384-gemma3-1b-pt_base-finetune"
    )
    if default_path.exists():
        return str(default_path)

    checkpoint_root = factory_root / "checkpoints" / "llava_factory"
    finetune_dirs = sorted(
        (
            path
            for path in checkpoint_root.glob("*-finetune")
            if path.is_dir()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if finetune_dirs:
        return str(finetune_dirs[0])

    return str(default_path)


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple TinyLLaVA inference entrypoint.")
    parser.add_argument("--model-path", type=str, default=default_model_path())
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--conv-mode", type=str, default="gemma3")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    eval_model(parse_args())
