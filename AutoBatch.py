#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Auto_BatchSize_API.py
import math
from typing import Optional, Union

def auto_batch(
    N: Optional[int] = None,
    alpha_or_level: Optional[Union[int, float]] = None,
    default: int = 128
) -> int:
    """
    Recommend an optimal batch size based on dataset size N and either:
    - no second arg → alpha = 0.5 (default behavior)
    - level 1~10 as int → alpha = level / 10

    Parameters:
        N (int, optional): Number of training samples
        alpha_or_level (int or float, optional):
            - None → default alpha = 0.5
            - int 1~10 → mapped to alpha = 0.1 ~ 1.0
        default (int): Fallback batch size if not computable

    Returns:
        int: Recommended batch size
    """
    if N is None:
        return default

    # Determine alpha from second argument
    if alpha_or_level is None:
        alpha = 0.5
    elif isinstance(alpha_or_level, int) and 1 <= alpha_or_level <= 10:
        alpha = alpha_or_level / 10.0
    else:
        raise ValueError("Second argument must be None or integer 1~10")

    assert N > 0, "N must be a positive integer"

    candidates = [2 ** i for i in range(5, 10)]  # 32 to 512
    raw = math.sqrt(N)
    scores = []
    for B in candidates:
        time = 1 / B
        gap = abs(B - raw) / raw
        score = alpha * time + (1 - alpha) * gap
        scores.append((score, B))

    _, best_batch = min(scores, key=lambda x: x[0])
    return best_batch


# In[ ]:




