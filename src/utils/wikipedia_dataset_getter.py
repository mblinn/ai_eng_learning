"""
Provides a getter for the Hugging Face Wikipedia dataset.
By default, fetches the first 500 elements, but allows parameterization.
"""
from typing import List, Dict, Any

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install the 'datasets' package: pip install datasets")

def get_wikipedia_items(num_items: int = 500, trust_remote_code: bool = True) -> List[Dict[str, Any]]:
    """
    Fetches the first `num_items` items from the Wikipedia dataset on Hugging Face.

    Args:
        num_items (int): Number of items to fetch. Defaults to 500.
        trust_remote_code (bool): Whether to trust remote code for the dataset. Defaults to True (required for wikipedia).

    Returns:
        List[Dict[str, Any]]: List of Wikipedia dataset items.
    """
    dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=trust_remote_code)
    return dataset.select(range(num_items))
