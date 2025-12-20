# Embedding cost configuration
from typing import Dict, Dict, List
from utils.logging import log

EMBEDDING_COST_PER_1K_TOKENS = 0.0001  # text-embedding-ada-002
AVG_CHARS_PER_TOKEN = 4               # ~4 chars per token (English)

# ===== LLM COST CONFIG =====

# OpenAI (example values – tune per model)
OPENAI_INPUT_COST_PER_1K_TOKENS = 0.0005
OPENAI_OUTPUT_COST_PER_1K_TOKENS = 0.0015

# Gemini Flash (approximate)
GEMINI_INPUT_COST_PER_1K_TOKENS = 0.00035
GEMINI_OUTPUT_COST_PER_1K_TOKENS = 0.00105

def estimate_llm_cost(
    prompt: str,
    response: str,
    model_name: str
) -> Dict[str, float]:
    """
    Estimate token usage and cost for LLM generation (Gemini / OpenAI).
    """
    input_chars = len(prompt)
    output_chars = len(response)

    input_tokens = input_chars / AVG_CHARS_PER_TOKEN
    output_tokens = output_chars / AVG_CHARS_PER_TOKEN

    if "gpt" in model_name.lower() or "openai" in model_name.lower():
        input_cost = (input_tokens / 1000) * OPENAI_INPUT_COST_PER_1K_TOKENS
        output_cost = (output_tokens / 1000) * OPENAI_OUTPUT_COST_PER_1K_TOKENS
        provider = "openai"
    else:
        input_cost = (input_tokens / 1000) * GEMINI_INPUT_COST_PER_1K_TOKENS
        output_cost = (output_tokens / 1000) * GEMINI_OUTPUT_COST_PER_1K_TOKENS
        provider = "gemini"
    log.info(
        f"[LLM COST ESTIMATION] ({provider}): "
        f"{int(input_tokens)} in / "
        f"{int(output_tokens)} out | "
        f"${round(input_cost + output_cost, 6):6f}"
    )
    return {
        "provider": provider,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }

def _estimate_embedding_cost(texts: List[str]) -> Dict[str, float]:
    """
    Estimate token usage and cost for embedding texts.
    """
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = total_chars / AVG_CHARS_PER_TOKEN
    estimated_cost = (estimated_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS

    return {
        "total_chars": total_chars,
        "estimated_tokens": int(estimated_tokens),
        "estimated_cost_usd": round(estimated_cost, 6),
    }
