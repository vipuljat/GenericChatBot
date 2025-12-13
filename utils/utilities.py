# Embedding cost configuration
from typing import Dict, Dict, List


EMBEDDING_COST_PER_1K_TOKENS = 0.0001  # text-embedding-ada-002
AVG_CHARS_PER_TOKEN = 4               # ~4 chars per token (English)

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
