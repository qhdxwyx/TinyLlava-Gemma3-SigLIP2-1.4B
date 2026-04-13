from transformers import AutoTokenizer, Gemma3ForCausalLM, GemmaForCausalLM

from . import register_llm

def _postprocess_gemma_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    return tokenizer

@register_llm('gemma')
def return_gemmaclass():
    return (GemmaForCausalLM, (AutoTokenizer, _postprocess_gemma_tokenizer))

# Register for Gemma-3 models which use gemma3_text model type.
@register_llm('gemma-3')
def return_gemma3class():
    return (Gemma3ForCausalLM, (AutoTokenizer, _postprocess_gemma_tokenizer))
