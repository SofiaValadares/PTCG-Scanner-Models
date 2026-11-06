"""Cartas do catálogo que não estão na leitura real, com erro no padrão do OCR."""

from __future__ import annotations

import random
import re
from collections import Counter

import pandas as pd

# Código impresso como texto a partir de Scarlet & Violet. SV é Supreme Victors.
TEXT_SETS = frozenset(
    {
        "SVI", "PAL", "OBF", "MEW", "PAR", "PAF", "TEF", "TWM", "SFA", "SCR",
        "SSP", "PRE", "JTG", "DRI", "WHT", "BLK", "MEG", "ASC", "PFL", "POR",
        "SVP", "SVE", "30C", "CRI", "PBL",
    }
)
SEED = 20261106
_CONFUSION = {
    "a": "o", "e": "c", "i": "l", "o": "a", "u": "v", "t": "l", "l": "i",
    "m": "n", "n": "m", "g": "c", "c": "e", "b": "h", "s": "5", "p": "r", "r": "n",
}


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _alnum(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()


def _number_parts(text: str):
    raw = re.sub(r"\s+", "", str(text or ""))
    match = re.search(r"([A-Za-z]*)(\d+)[/\-](\d+)", raw)
    if not match:
        return None
    return match.group(1).upper(), int(match.group(2)), int(match.group(3))


def _number_key(text: str) -> str:
    parts = _number_parts(text)
    if parts is None:
        return ""
    prefix, collector, total = parts
    return f"{prefix}{collector}/{total}"


def _card_key(set_id: str, number: str) -> tuple[str, str]:
    key = _number_key(number)
    return (_alnum(set_id), key or f"raw:{(number or '').strip()}")


def _pick(rng: random.Random, labels: list[str], weights: list[int]) -> str:
    roll = rng.randrange(sum(weights))
    passed = 0
    for label, weight in zip(labels, weights):
        passed += weight
        if roll < passed:
            return label
    return labels[-1]


def _mutate(text: str, rng: random.Random) -> str:
    chars = list(text)
    if not chars:
        return text
    index = rng.randrange(len(chars))
    source = chars[index]
    replacement = _CONFUSION.get(source.lower(), rng.choice("aeiourstn"))
    chars[index] = replacement.upper() if source.isupper() else replacement
    return "".join(chars)


def _printed_name(name_en: str, name_pt: str, rng: random.Random, pt_rate: float) -> str:
    if _fold(name_en) == _fold(name_pt) or not name_pt.strip():
        return name_en or name_pt
    if not name_en.strip():
        return name_pt
    return name_pt if rng.random() < pt_rate else name_en


def _noisy_name(name: str, rng: random.Random, labels: list[str], weights: list[int]) -> str:
    kind = _pick(rng, labels, weights)
    if kind == "empty" or not name.strip():
        return ""
    if kind == "exact":
        return name
    if kind == "contains":
        mode = rng.choice(("cut", "last", "extra"))
        if mode == "last" and " " in name.strip():
            return name.strip().split()[-1]
        if mode == "extra":
            return name + rng.choice(("X", "x", "PT", "EN"))
        cut = rng.randint(1, min(4, max(1, len(name) - 1)))
        return name[cut:] if rng.random() < 0.5 else name[:-cut]
    mutated = name
    for _ in range(4):
        mutated = _mutate(mutated, rng)
        folded = _fold(mutated)
        original = _fold(name)
        if folded and folded not in original and original not in folded:
            return mutated
    return mutated


def _format_number(prefix: str, collector: int, total: int) -> str:
    return f"{prefix}{max(1, collector)}/{max(1, total)}"


def _noisy_number(number: str, rng: random.Random, labels: list[str], weights: list[int], junk_rate: float, blank_rate: float) -> str:
    kind = _pick(rng, labels, weights)
    parts = _number_parts(number)
    if parts is None:
        return "" if kind == "no_number" else number
    prefix, collector, total = parts
    if kind == "no_number":
        if rng.random() < blank_rate:
            return ""
        token = _mutate(prefix or "SV", rng)
        return token[: rng.randint(2, max(2, len(token)))]
    if kind == "collector_only":
        shift = rng.choice((-8, -5, -3, -2, -1, 1, 2, 3, 5, 8))
        return _format_number(prefix, collector, total + shift)
    if kind == "total_only":
        shift = rng.choice((-5, -3, -2, -1, 1, 2, 3, 5))
        return _format_number(prefix, collector + shift, total)
    if kind == "near":
        return _format_number(prefix, collector + rng.choice((-2, -1, 1, 2)), total)
    if kind == "other":
        if rng.random() < 0.5:
            return f"{prefix}{collector}{total}"
        return _format_number(
            prefix,
            collector + rng.choice((-4, -3, 3, 4)),
            total + rng.choice((-6, -4, 4, 6)),
        )
    text = _format_number(prefix, collector, total)
    if rng.random() < junk_rate:
        text += rng.choice(("-", "."))
    return text


def _noisy_collection(set_id: str, rng: random.Random, profile: dict) -> str:
    if set_id not in TEXT_SETS:
        return ""
    kind = _pick(rng, profile["labels"], profile["weights"])
    if kind == "empty":
        return ""
    if kind == "exact":
        return set_id
    if kind == "prefix":
        if profile["suffixes"] and rng.random() > profile["truncate_rate"]:
            return set_id + rng.choices(profile["suffixes"], weights=profile["suffix_weights"], k=1)[0]
        return set_id[:-1] if len(set_id) > 2 else set_id
    text = _mutate(set_id, rng)
    if text == set_id:
        text = set_id[:-1] + _mutate(set_id[-1:], rng)
    if rng.random() < 0.55:
        text += rng.choice(("PT", "EN", "P", "T"))
    return text


def _patterns(real: pd.DataFrame) -> dict:
    name_counts = Counter()
    pt_only = en_only = 0
    number_counts = Counter()
    junk = parseable = 0
    blank = no_number = 0
    collection_counts = Counter()
    suffixes = Counter()
    truncated = 0
    for row in real.itertuples(index=False):
        ocr_name = _fold(row.ocr_name)
        en, pt = _fold(row.catalog_name_en), _fold(row.catalog_name_pt)
        if not ocr_name:
            name_counts["empty"] += 1
        elif ocr_name in {en, pt}:
            name_counts["exact"] += 1
            if ocr_name == pt and ocr_name != en:
                pt_only += 1
            elif ocr_name == en and ocr_name != pt:
                en_only += 1
        elif ocr_name and (ocr_name in en or ocr_name in pt or en in ocr_name or pt in ocr_name):
            name_counts["contains"] += 1
        else:
            name_counts["other"] += 1
        parts = _number_parts(row.ocr_number)
        catalog = _number_parts(row.catalog_number)
        if parts is None:
            number_counts["no_number"] += 1
            no_number += 1
            blank += int(not str(row.ocr_number).strip())
        elif catalog is None:
            number_counts["other"] += 1
        else:
            parseable += 1
            same_collector = parts[1] == catalog[1]
            same_total = parts[2] == catalog[2]
            near = abs(parts[1] - catalog[1]) <= 2 and not same_collector
            if same_collector and same_total and parts[0] == catalog[0]:
                number_counts["exact"] += 1
                raw = str(row.ocr_number).strip()
                junk += int(bool(re.search(r"[^A-Za-z0-9/\-\s]", raw) or raw.endswith(("-", "."))))
            elif same_collector and same_total:
                number_counts["exact"] += 1
            elif near and same_total:
                number_counts["near"] += 1
            elif same_collector:
                number_counts["collector_only"] += 1
            elif same_total:
                number_counts["total_only"] += 1
            else:
                number_counts["other"] += 1
        if row.catalog_set_id not in TEXT_SETS:
            continue
        read = _alnum(row.ocr_collection)
        code = _alnum(row.catalog_set_id)
        if not read:
            collection_counts["empty"] += 1
        elif read == code:
            collection_counts["exact"] += 1
        elif read.startswith(code):
            collection_counts["prefix"] += 1
            suffixes[read[len(code):]] += 1
        elif code.startswith(read):
            collection_counts["prefix"] += 1
            truncated += 1
        else:
            collection_counts["garbled"] += 1
    prefix_total = max(1, collection_counts["prefix"])
    return {
        "name_labels": ["exact", "contains", "other", "empty"],
        "name_weights": [name_counts[label] or 1 for label in ("exact", "contains", "other", "empty")],
        "pt_rate": pt_only / (pt_only + en_only) if pt_only + en_only else 0.63,
        "number_labels": ["exact", "no_number", "collector_only", "total_only", "other", "near"],
        "number_weights": [
            number_counts[label] or 1
            for label in ("exact", "no_number", "collector_only", "total_only", "other", "near")
        ],
        "junk_rate": junk / parseable if parseable else 0.05,
        "blank_rate": blank / no_number if no_number else 0.1,
        "collection": {
            "labels": ["empty", "exact", "prefix", "garbled"],
            "weights": [collection_counts[label] or 1 for label in ("empty", "exact", "prefix", "garbled")],
            "suffixes": list(suffixes),
            "suffix_weights": list(suffixes.values()),
            "truncate_rate": truncated / prefix_total,
        },
    }


def _choose(real: pd.DataFrame, catalog: pd.DataFrame, rng: random.Random) -> list:
    present = {
        _card_key(row.catalog_set_id, row.catalog_number)
        for row in real.itertuples(index=False)
        if str(row.catalog_set_id).strip()
    }
    present_sets = {row.catalog_set_id for row in real.itertuples(index=False) if str(row.catalog_set_id).strip()}
    missing_sets = []
    unseen = {set_id: [] for set_id in catalog["set_id"].unique() if set_id in present_sets}
    for row in catalog.itertuples(index=False):
        if _card_key(row.set_id, row.number) in present:
            continue
        if row.set_id in present_sets:
            unseen[row.set_id].append(row)
        else:
            missing_sets.append(row)
    rng.shuffle(missing_sets)
    for cards in unseen.values():
        rng.shuffle(cards)
    chosen = list(missing_sets)
    budget = len(real)
    while len(chosen) < budget and any(unseen.values()):
        available = [set_id for set_id, cards in unseen.items() if cards]
        weights = [2 if set_id in TEXT_SETS else 1 for set_id in available]
        set_id = rng.choices(available, weights=weights, k=1)[0]
        chosen.append(unseen[set_id].pop())
    # Cartas de SVI em diante são as que as pessoas mais têm. Entram todas as que ainda faltam.
    newer = [card for set_id, cards in unseen.items() if set_id in TEXT_SETS for card in cards]
    rng.shuffle(newer)
    chosen.extend(newer)
    return chosen


def expand_decision(real: pd.DataFrame, catalog: pd.DataFrame, rng: random.Random | None = None) -> pd.DataFrame:
    """Acrescenta cartas ausentes. De SVI em diante entram todas; a coleção sintética só existe aí."""
    rng = rng or random.Random(SEED)
    profile = _patterns(real)
    catalog = catalog.drop_duplicates(["set_id", "number"]).reset_index(drop=True)
    rows = []
    for card in _choose(real, catalog, rng):
        printed = _printed_name(card.name_en, card.name_pt, rng, profile["pt_rate"])
        rows.append(
            {
                "ocr_name": _noisy_name(printed, rng, profile["name_labels"], profile["name_weights"]),
                "ocr_number": _noisy_number(
                    card.number,
                    rng,
                    profile["number_labels"],
                    profile["number_weights"],
                    profile["junk_rate"],
                    profile["blank_rate"],
                ),
                "ocr_collection": _noisy_collection(card.set_id, rng, profile["collection"]),
                "gt_name": card.name_en or card.name_pt,
                "gt_number": card.number,
                "gt_set": card.set_id,
                "gt_raw": "",
                "catalog_number": card.number,
                "catalog_set_id": card.set_id,
                "catalog_name_en": card.name_en,
                "catalog_name_pt": card.name_pt,
                "catalog_set_name_en": card.set_name_en,
                "catalog_set_name_pt": card.set_name_pt,
                "in_catalog": 1,
                "origem": "sintetica",
            }
        )
    synthetic = pd.DataFrame(rows)
    if not synthetic.empty:
        synthetic = synthetic.sort_values(["catalog_set_id", "catalog_number"], kind="stable")
    base = real.copy()
    if "origem" not in base.columns:
        base["origem"] = "real"
    return pd.concat([base, synthetic], ignore_index=True)
