import os
from pathlib import Path
import sys

from transformers import AutoConfig


FACTORY_ROOT = Path(__file__).resolve().parents[3]
LOCAL_DATA_ROOT = FACTORY_ROOT.parent / "data"
LOCAL_THIRD_PARTY_ROOT = FACTORY_ROOT.parent / "third_party"
SERVER_DATA_ROOT = Path("/root/autodl-tmp/TinyLLaVA_Factory/data")
SERVER_THIRD_PARTY_ROOT = Path("/root/autodl-tmp/TinyLLaVA_Factory/third_party")

DATA_ROOT = Path(
    os.environ.get(
        "DATA_ROOT",
        str(SERVER_DATA_ROOT if SERVER_DATA_ROOT.exists() else LOCAL_DATA_ROOT),
    )
)
THIRD_PARTY_ROOT = Path(
    os.environ.get(
        "THIRD_PARTY_ROOT",
        str(SERVER_THIRD_PARTY_ROOT if SERVER_THIRD_PARTY_ROOT.exists() else LOCAL_THIRD_PARTY_ROOT),
    )
)

GEMMA_PATH = THIRD_PARTY_ROOT / "LLM-Research" / "gemma-3-1b-pt"
SIGLIP2_PATH = THIRD_PARTY_ROOT / "google" / "siglip2-so400m-patch14-384"
PRETRAIN_JSON = DATA_ROOT / "text_files" / "blip_laion_cc_sbu_558k.json"
FINETUNE_JSON = DATA_ROOT / "text_files" / "llava_v1_5_mix665k_cleaned_data_w_ocr_vqa.json"
PRETRAIN_IMAGE_DIR = DATA_ROOT / "llava" / "llava_pretrain" / "images"


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")


def warn_if_missing(path: Path) -> None:
    if not path.exists():
        print(f"[warn] Missing optional path on this machine: {path}")


def main() -> None:
    for path in (FACTORY_ROOT, GEMMA_PATH, SIGLIP2_PATH):
        assert_exists(path)
    for path in (DATA_ROOT, PRETRAIN_JSON, FINETUNE_JSON, PRETRAIN_IMAGE_DIR):
        warn_if_missing(path)

    gemma_cfg = AutoConfig.from_pretrained(GEMMA_PATH, local_files_only=True)
    siglip_cfg = AutoConfig.from_pretrained(SIGLIP2_PATH, local_files_only=True)
    assert getattr(gemma_cfg, "model_type", None) == "gemma3_text"
    assert getattr(siglip_cfg, "model_type", None) == "siglip"

    sys.path.insert(0, str(FACTORY_ROOT))

    template_check = "skipped"
    try:
        from tinyllava.data.template import TemplateFactory
        from tinyllava.model.llm import LLMFactory
        from tinyllava.model.vision_tower import VisionTowerFactory

        TemplateFactory("gemma3")()
        LLMFactory(str(GEMMA_PATH))
        VisionTowerFactory(str(SIGLIP2_PATH))
        template_check = "passed"
    except ModuleNotFoundError as exc:
        print(f"[warn] Python dependency missing for deep import validation: {exc}")

    print("[ok] GemmaLlava root:", FACTORY_ROOT)
    print("[ok] Gemma-3 text backbone:", GEMMA_PATH)
    print("[ok] SigLIP2 vision tower:", SIGLIP2_PATH)
    print("[info] Expected pretrain annotation:", PRETRAIN_JSON)
    print("[info] Expected finetune annotation:", FINETUNE_JSON)
    print(f"[info] Gemma3 template/factory import validation: {template_check}")


if __name__ == "__main__":
    main()
