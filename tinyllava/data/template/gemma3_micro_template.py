import copy
from dataclasses import dataclass

from . import register_template
from .base import Template
from .formatter import EmptyFormatter, Formatter, StringFormatter
from ...utils.constants import IGNORE_INDEX


MICRO_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
)


@register_template("gemma3_micro")
@dataclass
class Gemma3MicroTemplate(Template):
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(
        slot="<start_of_turn>user\n{{content}}<end_of_turn>\n"
    )
    format_assistant: "Formatter" = StringFormatter(
        slot="<start_of_turn>model\n{{content}}<end_of_turn>\n"
    )
    system: "Formatter" = EmptyFormatter(slot=MICRO_SYSTEM_PROMPT)
    separator: "Formatter" = EmptyFormatter(
        slot=["<start_of_turn>model\n", "<end_of_turn>\n"]
    )

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        sep, _ = self.separator.apply()
        user_sep = "<start_of_turn>user\n"
        prompt_parts = prompt.split(user_sep)
        prefix_prompt = prompt_parts[0] if prompt_parts else ""
        rounds = [user_sep + chunk for chunk in prompt_parts[1:] if chunk]

        cur_prompt = prefix_prompt
        cur_len = len(self.tokenizer_image_token(prefix_prompt, tokenizer))
        if cur_len > 0:
            labels[:cur_len] = IGNORE_INDEX

        for rou in rounds:
            parts = rou.split(sep, 1)
            if len(parts) != 2:
                break
            instruction = parts[0] + sep
            instruction_end = len(
                self.tokenizer_image_token(cur_prompt + instruction, tokenizer)
            )
            round_end = len(self.tokenizer_image_token(cur_prompt + rou, tokenizer))
            labels[cur_len:instruction_end] = IGNORE_INDEX
            cur_len = round_end
            cur_prompt += rou

        labels[cur_len:] = IGNORE_INDEX
        return labels
