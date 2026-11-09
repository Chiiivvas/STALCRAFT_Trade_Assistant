import re
from difflib import SequenceMatcher

_QUANTITY_RE = re.compile(r"(?i)[xх×]\s*[:=.,-]?\s*(\d{1,6})")


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


def offer_anchor_seen(lines):
    for line in lines:
        text = normalize_text(line["text"])
        if "вы получите" in text or "вы получаете" in text:
            return True
        if SequenceMatcher(None, text.replace(" ", ""), "выполучите").ratio() >= 0.88:
            return True
    return False


def _match_price_name(candidate, normalized_names):
    normalized_candidate = normalize_text(candidate)
    if not normalized_candidate:
        return None
    for normalized_name, original_name in sorted(
        normalized_names, key=lambda pair: len(pair[0]), reverse=True
    ):
        if normalized_name and (
            normalized_name in normalized_candidate
            or normalized_candidate in normalized_name
        ):
            return original_name
    best_name = None
    best_score = 0.0
    for normalized_name, original_name in normalized_names:
        score = SequenceMatcher(None, normalized_candidate, normalized_name).ratio()
        if score > best_score:
            best_score = score
            best_name = original_name
    return best_name if best_score >= 0.84 else None


def parse_offer_lines(lines, price_names):
    """Parse explicit xN rows below «Вы получите», including unpriced names."""
    anchor_y = None
    for line in lines:
        if offer_anchor_seen([line]):
            anchor_y = line["y"]
            break
    if anchor_y is None:
        return {}

    normalized_names = [
        (normalize_text(name), name)
        for name in price_names if normalize_text(name)
    ]
    totals = {}
    for line in lines:
        if line["y"] <= anchor_y:
            continue
        text = line["text"]
        quantity_match = _QUANTITY_RE.search(text)
        if not quantity_match:
            continue
        quantity = int(quantity_match.group(1))
        if quantity < 1:
            continue

        candidate = _QUANTITY_RE.sub("", text)
        candidate = re.sub(r"[|丨Il]{2,}", " ", candidate)
        candidate = " ".join(candidate.split()).strip(" -:|")
        if not normalize_text(candidate):
            continue

        matched_name = _match_price_name(candidate, normalized_names)
        display_name = matched_name or candidate
        totals[display_name] = totals.get(display_name, 0) + quantity
    return totals


def _sidebar_visual_rows(lines):
    """Group merged OCR tokens into visual rows and repair split stack digits."""
    words = [
        word
        for line in lines
        for word in line.get("words", [])
        if word.get("text", "").strip()
    ]
    if not words:
        return [line.get("text", "") for line in lines]

    heights = sorted(max(1, word.get("height", 1)) for word in words)
    median_height = heights[len(heights) // 2]
    row_tolerance = max(5, median_height * 0.72)
    words.sort(key=lambda word: (
        word["y"] + word.get("height", 0) / 2, word["x"]
    ))

    rows = []
    for word in words:
        center_y = word["y"] + word.get("height", 0) / 2
        best_row = None
        best_distance = None
        for row in rows[-3:]:
            row_center = sum(
                item["y"] + item.get("height", 0) / 2 for item in row
            ) / len(row)
            distance = abs(center_y - row_center)
            if distance <= row_tolerance and (
                best_distance is None or distance < best_distance
            ):
                best_row, best_distance = row, distance
        if best_row is None:
            rows.append([word])
        else:
            best_row.append(word)

    rendered = []
    for row in rows:
        row.sort(key=lambda item: item["x"])
        tokens = []
        for word in row:
            token = word["text"].strip()
            if not token:
                continue
            if tokens:
                previous = tokens[-1]
                prev_right = previous["x"] + previous["width"]
                gap = word["x"] - prev_right
                # Tesseract occasionally splits a stack count into e.g. "13" + "4".
                if (previous["text"].isdigit() and token.isdigit()
                        and gap <= max(8, median_height * 0.55)):
                    previous["text"] += token
                    previous["width"] = (
                        word["x"] + word.get("width", 1) - previous["x"]
                    )
                    continue
            tokens.append({
                "text": token,
                "x": word["x"],
                "width": word.get("width", 1)
            })
        rendered.append(" ".join(token["text"] for token in tokens))
    return rendered


def parse_sidebar_offer_lines(lines, price_names):
    """Parse each sidebar item row independently, preserving unpriced names."""
    normalized_names = [
        (normalize_text(name), name)
        for name in price_names if normalize_text(name)
    ]
    totals = {}

    for text in _sidebar_visual_rows(lines):
        text = text.replace("Х", "х").replace("X", "x")
        quantity_match = _QUANTITY_RE.search(text)
        if not quantity_match:
            continue

        quantity = int(quantity_match.group(1))
        if quantity < 1:
            continue

        candidate = _QUANTITY_RE.sub("", text)
        candidate = re.sub(r"[|丨Il]{2,}", " ", candidate)
        candidate = " ".join(candidate.split()).strip(" -:|")
        if not normalize_text(candidate):
            continue

        matched_name = _match_price_name(candidate, normalized_names)
        display_name = matched_name or candidate
        totals[display_name] = totals.get(display_name, 0) + quantity

    return totals
