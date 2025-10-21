try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formatting_utils import formatting_prompts_func
import json

repo_root = Path(__file__).resolve().parents[1]
DATA = repo_root / "data" / "train_chat_en.json"


def preview(n=5):
    if not DATA.exists():
        print(f"Data file not found: {DATA}")
        return

    if load_dataset is not None:
        ds = load_dataset("json", data_files=str(DATA), split="train")
        iterator = enumerate(ds)
    else:
        # Fallback: read JSON lines
        with open(DATA, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # try to parse as JSONL (one JSON object per line)
        items = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                # try to handle a file that is a JSON array
                try:
                    items = json.loads(content)
                    break
                except Exception:
                    raise
        iterator = enumerate(items)

    for i, ex in iterator:
        if i >= n:
            break
        print("--- Example", i, "---")
        out = formatting_prompts_func(ex)
        print("Formatted output (list length):", len(out))
        for j, s in enumerate(out):
            print(f"[{j}]", s)


if __name__ == '__main__':
    preview(5)
