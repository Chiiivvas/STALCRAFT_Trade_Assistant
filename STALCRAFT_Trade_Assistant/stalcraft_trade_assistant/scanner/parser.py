import re


def group_words_into_lines(words):
    grouped = {}
    for word in words:
        grouped.setdefault(word["line"], []).append(word)

    lines = []
    for items in grouped.values():
        items.sort(key=lambda item: item["x"])
        lines.append({
            "text": " ".join(item["text"] for item in items),
            "y": min(item["y"] for item in items),
            "words": items
        })
    return sorted(lines, key=lambda item: item["y"])


def normalize_text(value):
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^а-яa-z0-9]+", " ", value)
    return " ".join(value.split())


def parse_offer_lines(lines, price_names):
    """
    Read only item rows below the UI caption «Вы получите».
    Each row must contain a known price-list name and an explicit xN count.
    If the caption or count is missing, do not guess or count the item.
    """
    anchor_y = None
    for line in lines:
        text = normalize_text(line["text"])
        if "вы получите" in text or "вы получаете" in text:
            anchor_y = line["y"]
            break

    if anchor_y is None:
        return {}

    normalized_names = [
        (normalize_text(name), name)
        for name in price_names
        if normalize_text(name)
    ]
    totals = {}

    for line in lines:
        if line["y"] <= anchor_y:
            continue

        text = line["text"]
        normalized_line = normalize_text(text)
        matched_name = None

        # Prefer the longest exact normalized phrase contained in this OCR row.
        for normalized_name, original_name in sorted(
            normalized_names, key=lambda pair: len(pair[0]), reverse=True
        ):
            if normalized_name and normalized_name in normalized_line:
                matched_name = original_name
                break

        if not matched_name:
            continue

        quantity_match = re.search(r"(?i)[xх×]\s*(\d{1,6})", text)
        if not quantity_match:
            continue

        quantity = int(quantity_match.group(1))
        if quantity < 1:
            continue

        totals[matched_name] = totals.get(matched_name, 0) + quantity

    return totals
