# VLMEvalKit for TinyLlava-Gemma-SigLIP2-1.4B

This directory contains the local `VLMEvalKit` config for evaluating the finetuned TinyLlava-Gemma-SigLIP2-1.4B model.

## Files

- `gemmallava_local.json`: sample config for the current finetuned checkpoint
- `compare_template.json`: sample multi-model comparison config

## Recommended first-pass benchmarks

- `MME`
- `MMVet`
- `TextVQA_VAL`
- `GQA_TestDev_Balanced`
- `MMBench_V11`
- `MMMU_DEV_VAL`

These are all natively supported by the current `VLMEvalKit` checkout and are good overlap benchmarks for comparing against mainstream MLLMs.

## Run on the current local checkpoint

From the workspace root:

```powershell
conda activate TinyLlava
$env:LMUData = "E:\LLM_projects\GemmaLlava\VLMEvalKit\LMUData"
New-Item -ItemType Directory -Force -Path $env:LMUData | Out-Null
cd E:\LLM_projects\GemmaLlava\VLMEvalKit
python run.py --config ..\TinyLlava-Gemma-SigLIP2-1.4B\scripts\eval\vlmeval\gemmallava_local.json --work-dir ..\TinyLlava-Gemma-SigLIP2-1.4B\outputs\vlmeval
```

## Run only a subset of datasets

```powershell
conda activate TinyLlava
$env:LMUData = "E:\LLM_projects\GemmaLlava\VLMEvalKit\LMUData"
New-Item -ItemType Directory -Force -Path $env:LMUData | Out-Null
cd E:\LLM_projects\GemmaLlava\VLMEvalKit
python run.py --config ..\TinyLlava-Gemma-SigLIP2-1.4B\scripts\eval\vlmeval\gemmallava_local.json --data MME MMVet TextVQA_VAL --work-dir ..\TinyLlava-Gemma-SigLIP2-1.4B\outputs\vlmeval
```

## Run MMBench TEST EN V11 only

Download and place this file at:

```text
E:\LLM_projects\GemmaLlava\VLMEvalKit\LMUData\MMBench_TEST_EN_V11.tsv
```

Dataset URL:

```text
https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN_V11.tsv
```

Run:

```powershell
conda activate TinyLlava
$env:LMUData = "E:\LLM_projects\GemmaLlava\VLMEvalKit\LMUData"
New-Item -ItemType Directory -Force -Path $env:LMUData | Out-Null
cd E:\LLM_projects\GemmaLlava\VLMEvalKit
python run.py --config ..\TinyLlava-Gemma-SigLIP2-1.4B\scripts\eval\vlmeval\gemmallava_local.json --data MMBench_TEST_EN_V11 --work-dir ..\TinyLlava-Gemma-SigLIP2-1.4B\outputs\vlmeval
```

## Run MMBench_V11

```powershell
conda activate TinyLlava
$env:LMUData = "E:\LLM_projects\GemmaLlava\VLMEvalKit\LMUData"
New-Item -ItemType Directory -Force -Path $env:LMUData | Out-Null
cd E:\LLM_projects\GemmaLlava\VLMEvalKit
python run.py --config ..\TinyLlava-Gemma-SigLIP2-1.4B\scripts\eval\vlmeval\gemmallava_local.json --data MMBench_V11 --work-dir ..\TinyLlava-Gemma-SigLIP2-1.4B\outputs\vlmeval
```

## Common comparison models already supported by VLMEvalKit

- `Qwen2.5-VL-7B-Instruct`
- `Qwen2-VL-7B-Instruct`
- `llava_v1.5_7b`
- `llava_v1.5_13b`
- `llava_next_llama3`
- `GeminiFlash2-5`
- `GeminiPro2-5`

## Compare multiple models with one config

```powershell
conda activate TinyLlava
$env:LMUData = "E:\LLM_projects\GemmaLlava\VLMEvalKit\LMUData"
New-Item -ItemType Directory -Force -Path $env:LMUData | Out-Null
cd E:\LLM_projects\GemmaLlava\VLMEvalKit
python run.py --config ..\TinyLlava-Gemma-SigLIP2-1.4B\scripts\eval\vlmeval\compare_template.json --work-dir ..\TinyLlava-Gemma-SigLIP2-1.4B\outputs\vlmeval_compare
```

## Notes

- `VQAv2 test-dev` is not part of the default benchmark set in the current `VLMEvalKit` checkout. If needed, add it as a custom dataset class or keep using the existing TinyLLaVA evaluation path for that specific benchmark.
- API models such as Gemini require the corresponding API key in `VLMEvalKit/.env` or environment variables.
- `MMBench_V11` is supported in the current checkout. It is an MCQ benchmark, so it can run without an API judge, but answer extraction is usually more robust if `OPENAI_API_KEY` is configured.
