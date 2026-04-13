import copy
from dataclasses import dataclass

from . import register_template
from .base import Template
from .formatter import EmptyFormatter, Formatter, StringFormatter
from ...utils.constants import IGNORE_INDEX


@register_template("gemma3")
@dataclass
class Gemma3Template(Template):
    format_image_token: "Formatter" = StringFormatter(slot="<image>\n{{content}}")
    format_user: "Formatter" = StringFormatter(
        slot="<start_of_turn>user\n{{content}}<end_of_turn>\n"
    )
    format_assistant: "Formatter" = StringFormatter(
        slot="<start_of_turn>model\n{{content}}<end_of_turn>\n"
    )
    system: "Formatter" = EmptyFormatter(slot="")
    separator: "Formatter" = EmptyFormatter(
        slot=["<start_of_turn>model\n", "<end_of_turn>\n"]
    )

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        sep, _ = self.separator.apply()
        user_sep = "<start_of_turn>user\n"
        cur_prompt = ""
        cur_len = (
            1
            if len(input_ids) > 0 and getattr(tokenizer, "bos_token_id", None) is not None and input_ids[0] == tokenizer.bos_token_id
            else 0
        )
        if cur_len > 0:
            labels[:cur_len] = IGNORE_INDEX

        rounds = [user_sep + chunk for chunk in prompt.split(user_sep) if chunk]
        for rou in rounds:
            parts = rou.split(sep, 1)
            if len(parts) != 2:
                break
            instruction = parts[0] + sep
            instruction_end = len(self.tokenizer_image_token(cur_prompt + instruction, tokenizer))
            round_end = len(self.tokenizer_image_token(cur_prompt + rou, tokenizer))
            labels[cur_len:instruction_end] = IGNORE_INDEX
            cur_len = round_end
            cur_prompt += rou

        labels[cur_len:] = IGNORE_INDEX
        return labels
