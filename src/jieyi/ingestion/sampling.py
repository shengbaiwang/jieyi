from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DistributedSample:
    text: str
    excerpt_count: int
    source_chars: int


def _snap_forward(text: str, position: int, distance: int = 60) -> int:
    if position <= 0:
        return 0
    for index in range(position, min(len(text), position + distance)):
        if text[index].isspace():
            return index + 1
    return position


def _snap_backward(text: str, position: int, distance: int = 60) -> int:
    if position >= len(text):
        return len(text)
    for index in range(position, max(0, position - distance), -1):
        if text[index - 1].isspace():
            return index - 1
    return position


def take_distributed_sample(
    text: str,
    *,
    total_budget: int,
    excerpt_count: int = 8,
    minimum_excerpt_chars: int = 200,
    separator: str = "\n\n[… omitted …]\n\n",
) -> DistributedSample:
    """Sample evenly across a long document while respecting a hard character budget."""
    if total_budget <= 0 or not text:
        return DistributedSample("", 0, len(text))
    if len(text) <= total_budget:
        return DistributedSample(text, 1, len(text))

    count = max(1, excerpt_count)
    count = min(count, max(1, total_budget // max(1, minimum_excerpt_chars)))
    if count == 1:
        return DistributedSample(text[:total_budget], 1, len(text))

    separator_cost = len(separator) * (count - 1)
    usable_budget = max(count, total_budget - separator_cost)
    window = max(1, usable_budget // count)
    last_start = max(0, len(text) - window)
    starts = [round(index * last_start / (count - 1)) for index in range(count)]

    excerpts: list[str] = []
    previous_end = 0
    for raw_start in starts:
        start = _snap_forward(text, raw_start)
        start = max(start, previous_end)
        end = _snap_backward(text, min(len(text), start + window))
        if end <= start:
            end = min(len(text), start + window)
        excerpt = text[start:end].strip()
        if excerpt:
            excerpts.append(excerpt)
            previous_end = end

    joined = separator.join(excerpts)
    if len(joined) > total_budget:
        joined = joined[:total_budget]
    return DistributedSample(joined, len(excerpts), len(text))

