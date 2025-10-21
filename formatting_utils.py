def formatting_prompts_func(example):
    """Safely format various example shapes into a list containing
    a single processed string.

    Supports:
    - example as a list of message dicts
    - example as a dict with a 'messages' key
    - example as a dict with a 'text' key
    - fallback: if a dict contains any list-like value, use that
    """
    # Optional logging for malformed examples
    messages = example
    try:
        from pathlib import Path
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(exist_ok=True)
        bad_log = log_dir / "formatter_bad_examples.log"
    except Exception:
        bad_log = None
    if isinstance(example, dict):
        if "messages" in example and isinstance(example["messages"], list):
            messages = example["messages"]
        elif "text" in example:
            return [example["text"]]
        else:
            for v in example.values():
                if isinstance(v, list):
                    messages = v
                    break

    if not isinstance(messages, list):
        if bad_log is not None:
            try:
                with open(bad_log, "a", encoding="utf-8") as f:
                    f.write("NON_LIST_EXAMPLE:\n")
                    f.write(repr(example) + "\n---\n")
            except Exception:
                pass
        return [str(messages)]

    lines = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "Unknown").capitalize()
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        else:
            # Log non-dict message items for debugging
            if bad_log is not None:
                try:
                    with open(bad_log, "a", encoding="utf-8") as f:
                        f.write("NON_DICT_MESSAGE_ITEM:\n")
                        f.write(repr(m) + "\n---\n")
                except Exception:
                    pass
            lines.append(str(m))

    return ["\n".join(lines)]
