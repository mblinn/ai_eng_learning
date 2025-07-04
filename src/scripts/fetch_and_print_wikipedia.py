"""
Fetches and prints the first 10 elements from the Wikipedia Hugging Face dataset.
"""
import sys
from utils.wikipedia_dataset_getter import get_wikipedia_items

if __name__ == "__main__":
    print("[INFO] Script started", file=sys.stderr)
    try:
        items = get_wikipedia_items(10)
        print(f"[INFO] Successfully fetched {len(items)} items", file=sys.stderr)
        if len(items) > 0:
            print("[INFO] Sample item:", file=sys.stderr)
            print(items[0], file=sys.stderr)
        else:
            print("[WARNING] No items fetched.", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}", file=sys.stderr)
        raise
    for i, item in enumerate(items):
        print(f"Item {i+1}:")
        print(item)
        print("-" * 40)
