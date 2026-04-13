import argparse
import os
import requests
from PIL import Image
from io import BytesIO

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from transformers import PreTrainedModel

try:
    import torch._dynamo
    torch._dynamo.config.disable = True
except Exception:
    pass

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def resolve_inference_conv_mode(model, requested_conv_mode):
    if requested_conv_mode:
        return requested_conv_mode

    config_conv_mode = getattr(model.config, "conv_version", None)
    if config_conv_mode == "gemma3":
        return "gemma3_micro"
    if config_conv_mode:
        return config_conv_mode
    return "gemma3"


def resolve_max_new_tokens(model, context_len, input_ids, images_tensor, requested_max_new_tokens):
    max_context_length = getattr(model.config, "tokenizer_model_max_length", context_len)
    if requested_max_new_tokens is not None and requested_max_new_tokens > 0:
        return requested_max_new_tokens

    with torch.inference_mode():
        image_token_count = model.encode_images(images_tensor).shape[1]

    return max(1, max_context_length - input_ids.shape[1] - image_token_count)


def eval_model(args):
    # Model
    disable_torch_init()

    if args.model_path is not None:
        model, tokenizer, image_processor, context_len = load_pretrained_model(
            args.model_path, device=args.device
        )
    else:
        assert args.model is not None, 'model_path or model must be provided'
        model = args.model
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        tokenizer = model.tokenizer
        image_processor = model.vision_tower._image_processor
    conv_mode = resolve_inference_conv_mode(model, args.conv_mode)
    qs = args.query.strip()
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    text_processor = TextPreprocess(tokenizer, conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    model.to(args.device)

    msg = Message()
    msg.add_message(qs)

    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids']
    prompt = result['prompt']
    input_ids = input_ids.unsqueeze(0).to(model.device)
        

    image_files = image_parser(args)
    images = load_images(image_files)[0]
    images_tensor = image_processor(images)
    image_dtype = getattr(model, "dtype", torch.float16)
    images_tensor = images_tensor.unsqueeze(0).to(model.device, dtype=image_dtype)

    

    stop_str = text_processor.template.separator.apply()[1]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = args.temperature > 0
    max_new_tokens = resolve_max_new_tokens(
        model, context_len, input_ids, images_tensor, args.max_new_tokens
    )

    generate_kwargs = dict(
        images=images_tensor,
        do_sample=do_sample,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )
    if do_sample:
        generate_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            generate_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generate_kwargs)

    assistant_prefix = text_processor.template.separator.apply()[0]
    full_outputs = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    if assistant_prefix in full_outputs:
        outputs = full_outputs.rsplit(assistant_prefix, 1)[-1]
    else:
        outputs = full_outputs

    for suffix in (stop_str, stop_str.rstrip(), "<end_of_turn>", tokenizer.eos_token or ""):
        if suffix and suffix in outputs:
            outputs = outputs.split(suffix, 1)[0]
    outputs = outputs.replace(tokenizer.bos_token or "", "").strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=PreTrainedModel, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    args = parser.parse_args()

    eval_model(args)
