# TinyLlava-Gemma3-SigLIP2-1.4B

TinyLlava-Gemma3-SigLIP2-1.4B is a compact vision-language model project built on top of TinyLLaVA, using `gemma-3-1b-pt` as the language backbone and `siglip2-so400m-patch14-384` as the vision tower. This repository contains the training, finetuning, inference, and evaluation code for this Gemma3 + SigLIP2 variant.

The repository is designed for practical small-scale VLM training, especially single-GPU or lightweight multi-GPU workflows, with a simple two-stage pipeline:

- connector pretraining for multimodal alignment
- instruction finetuning for conversational multimodal ability

## Project Overview

- Base LLM: `gemma-3-1b-pt`
- Vision encoder: `google/siglip2-so400m-patch14-384`
- Connector: `mlp2x_gelu`
- Conversation template: `gemma3`
- Framework base: TinyLLaVA / TinyLLaVA Factory
- Model scale: roughly 1.4B parameters as a compact multimodal model
- Training hardware: `2 x RTX 5090 GPU`
- Experiment tracking: `SwanLab`

### Example Inference

Prompt: `Describe this figure.`

Example 1

![TEST1](./tinyllava/serve/examples/TEST1.jpg)

Output:

```text
The image features a tall, green and red building with a prominent spire on top. The tower is surrounded by a beautiful backdrop of trees and mountains in the distance. The scene appears to be set against a serene sky, creating an enchanting atmosphere.

In addition to the towering structure, there are several smaller buildings visible in the background, adding depth to the landscape. A few people can also be seen scattered throughout the area, likely admiring the impressive architecture or enjoying the view.
```

Example 2

![TEST2](./tinyllava/serve/examples/TEST2.jpg)

Output:

```text
The image features a large wicker basket filled with an assortment of fresh fruits. The basket is overflowing, showcasing a variety of apples and oranges placed in different positions within the basket. There are at least 13 apples and 4 oranges visible in the scene, creating a vibrant and colorful display.

In addition to the fruit, there are also grapes scattered throughout the basket, adding to the diverse selection of produce. This arrangement creates an appealing and inviting presentation for anyone who sees it.
```

## Environment Setup
```bash
git clone https://github.com/qhdxwyx/TinyLlava-Gemma3-SigLIP2-1.4B
cd TinyLlava-Gemma3-SigLIP2-1.4B

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install flash-attn --no-build-isolation
```

Recommended runtime notes:

- `transformers >= 4.50.0`
- CUDA environment with `bf16` support
- the default scripts can run on a single GPU

## Data Layout

The Gemma3 training scripts automatically resolve either a local layout or the development server layout.

Default local roots:

- `../data`
- `../third_party`

Default paths used by the training pipeline:

- LLM: `../third_party/LLM-Research/gemma-3-1b-pt`
- Vision tower: `../third_party/google/siglip2-so400m-patch14-384`
- Pretrain annotations: `../data/text_files/blip_laion_cc_sbu_558k.json`
- Pretrain images: `../data/llava/llava_pretrain/images`
- Finetune annotations: `../data/text_files/llava_v1_5_mix665k_cleaned_data_w_ocr_vqa.json`
- Finetune images: `../data`

Before training, run:

```bash
python scripts/train/gemma3/verify_local_setup.py
```

## Training Strategy

This repository follows a two-stage training recipe implemented under `scripts/train/gemma3/`.

### Stage 1: Connector Pretraining

The pretraining stage aligns visual representations from SigLIP2 with the Gemma3 language backbone before instruction tuning.

- Gemma3 LLM is frozen
- the MLP connector is trained fully
- the vision tower is partially tuned instead of fully updated
- only later vision layers are opened for training

Default settings in `pretrain_gemma3.sh`:

- `conv_version=pretrain`
- `tune_type_llm=frozen`
- `tune_type_connector=full`
- `tune_type_vision_tower=partially-tune`
- `tune_vision_tower_from_layer=20`
- learning rate: `3e-4`
- vision tower learning rate: `4.5e-5`
- per-device batch size: `4`
- gradient accumulation: `16`

### Stage 2: Full Finetuning

The finetuning stage instruction-tunes the model with the pretrained connector checkpoint.
- the stage-1 pretrained checkpoint is reused as initialization
- Gemma3 LLM is fully tuned
- the MLP connector continues full training
- the vision tower is frozen during this stage

Default settings in `finetune_gemma3.sh`:

- `conv_version=gemma3`
- `tune_type_llm=full`
- `tune_type_connector=full`
- `tune_type_vision_tower=frozen`
- pretrained stage-1 checkpoint reused as initialization
- learning rate: `1e-4`
- per-device batch size: `4`
- gradient accumulation: `8`

This recipe keeps the first stage focused on visual-language alignment and the second stage focused on conversational multimodal performance.

## Training

Run the full pipeline:

```bash
bash scripts/train/gemma3/train_gemma3_base.sh
```

Run only pretraining:

```bash
RUN_PRETRAIN=1 RUN_FINETUNE=0 bash scripts/train/gemma3/train_gemma3_base.sh
```

Run only finetuning:

```bash
RUN_PRETRAIN=0 RUN_FINETUNE=1 bash scripts/train/gemma3/train_gemma3_base.sh
```

Useful overrides:

```bash
VERSION=my_run \
MODEL_MAX_LENGTH=3072 \
PRETRAIN_BATCH_SIZE=8 \
FINETUNE_BATCH_SIZE=4 \
REPORT_TO=wandb \
bash scripts/train/gemma3/train_gemma3_base.sh
```

Training logs are tracked with `SwanLab`. This repository provides a lightweight `wandb.py` compatibility shim, so existing `REPORT_TO=wandb` workflows are redirected to `SwanLab` automatically.

Common environment variables:

- `DATA_ROOT`
- `THIRD_PARTY_ROOT`
- `LLM_VERSION`
- `VT_VERSION`
- `VERSION`
- `MODEL_MAX_LENGTH`
- `PRETRAIN_BATCH_SIZE`
- `PRETRAIN_GRAD_ACCUM`
- `FINETUNE_BATCH_SIZE`
- `FINETUNE_GRAD_ACCUM`
- `REPORT_TO`

## Inference

Bundled test images are available in `tinyllava/serve/examples/`.

Simple Python inference on Windows PowerShell:

```powershell
cd E:\LLM_projects\GemmaLlava\TinyLlava-Gemma3-SigLIP2-1.4B

python .\simply_inference.py `
  --image-file ".\tinyllava\serve\examples\TEST1.jpg" `
  --temperature 0 `
  --query "Describe this figure."
```

Simple Python inference on Linux/macOS bash:

```bash
python simply_inference.py \
  --image-file tinyllava/serve/examples/TEST1.jpg \
  --temperature 0 \
  --query "Describe this figure."
```

`simply_inference.py` will first try the default Gemma3 finetuned checkpoint path. If that folder does not exist, it will automatically pick the most recently modified `checkpoints/llava_factory/*-finetune` directory. You can also set `--model-path` explicitly.

You do not need to pass `--device` for normal local testing. The script will use `cuda:0` automatically when CUDA is available, and fall back to `cpu` otherwise. Only pass `--device` when you want to force a different GPU or CPU.

The default `temperature` is `0`. Increase it only when you want more diverse generations, for example `--temperature 0.1`.

## Evaluation

Evaluation on the `VQAv2` test set is currently in progress. **The current random-sampled score is about 72%.**

Next, we plan to evaluate with `VLMEvalKit`.

For more details, see:

- `scripts/eval/vlmeval/README.md`

## Acknowledgement

This project builds on TinyLLaVA / TinyLLaVA Factory and adapts it for a Gemma3 + SigLIP2 compact multimodal setup. Thanks to the TinyLLaVA authors and the open-source ecosystem behind Gemma3, SigLIP2, PyTorch, and Hugging Face.

## Citation
```bibtex
@misc{wang2024microllama,
  title        = {MicroLLaVA: a TinyLLaVA based VLM with MicroLlama 300M for single GPU training},
  author       = {Zixiao Ken Wang},
  year         = {2025},
  url          = {https://huggingface.co/keeeeenw/MicroLlava,https://huggingface.co/keeeeenw/MicroLlava-Qwen3-0.6B-base-siglip2-so400m}
}

@misc{zhou2024tinyllava,
      title={TinyLLaVA: A Framework of Small-scale Large Multimodal Models},
      author={Baichuan Zhou and Ying Hu and Xi Weng and Junlong Jia and Jie Luo and Xien Liu and Ji Wu and Lei Huang},
      year={2024},
      eprint={2402.14289},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@article{jia2024tinyllava,
  title={TinyLLaVA Factory: A Modularized Codebase for Small-scale Large Multimodal Models},
  author={Jia, Junlong and Hu, Ying and Weng, Xi and Shi, Yiming and Li, Miao and Zhang, Xingjian and Zhou, Baichuan and Liu, Ziyu and Luo, Jie and Huang, Lei and Wu, Ji},
  journal={arXiv preprint arXiv:2405.11788},
  year={2024}
}
```
