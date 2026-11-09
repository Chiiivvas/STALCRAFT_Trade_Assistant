import difflib
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


def _normalize(value):
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[^а-яa-z0-9]+", " ", value)
    return " ".join(value.split())


def _match_price_name(candidate, price_names):
    normalized = _normalize(candidate)
    if not normalized:
        return None

    normalized_names = {_normalize(name): name for name in price_names}
    if normalized in normalized_names:
        return normalized_names[normalized]

    matches = difflib.get_close_matches(
        normalized, list(normalized_names.keys()), n=1, cutoff=0.62
    )
    return normalized_names[matches[0]] if matches else None


def parse_offer_lines(lines, price_names):
    """
    Extract item names and stack counts only below the «Вы получите» label.
    A missing anchor intentionally returns no items rather than reading the
    other player's or user's unrelated grid.
    """
    anchor_y = None
    for line in lines:
        text = _normalize(line["text"])
        if "вы получите" in text or "вы получаете" in text:
            anchor_y = line["y"]
            break

    if anchor_y is None:
        return {}

    totals = {}
    for line in lines:
        if line["y"] <= anchor_y:
            continue

        text = line["text"].strip()
        quantity = re.search(r"(?i)[xх×]\s*(\d{1,6})", text)
        if not quantity:
            continue

        candidate = text[:quantity.start()].strip(" .,:;|[]()_?-")
        name = _match_price_name(candidate, price_names)
        if not name:
            continue

        qty = int(quantity.group(1))
        if qty > 0:
            totals[name] = totals.get(name, 0) + qty

    return totals
