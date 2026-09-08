export type ChapterLocation = {
  id: string;
  level: number;
  start_ordinal: number;
  end_ordinal: number;
  target?: { format: "pdf"; page: number };
  shared_range?: boolean;
};

// Identity is independent of position: source bookmarks can share a destination.
export function activeChapter<T extends ChapterLocation>(
  chapters: T[], ordinal: number, preferredId?: string | null,
): T | null {
  const candidates = chapters.filter((item) => ordinal >= item.start_ordinal && ordinal <= item.end_ordinal);
  const preferred = candidates.find((item) => item.id === preferredId);
  if (preferred) return preferred;
  return candidates.reduce<T | null>((best, item) => !best
    || item.start_ordinal > best.start_ordinal
    || (item.start_ordinal === best.start_ordinal && item.level > best.level) ? item : best, null);
}

export function toggleChapter(ids: string[], id: string): string[] {
  return ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id];
}

export function normalizeSegmentRanges(ranges: [number, number][]) {
  const sorted = ranges
    .map(([start, end]) => [Math.min(start, end), Math.max(start, end)] as [number, number])
    .sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  const merged: [number, number][] = [];
  for (const [start, end] of sorted) {
    const previous = merged.at(-1);
    if (!previous || start > previous[1] + 1) merged.push([start, end]);
    else previous[1] = Math.max(previous[1], end);
  }
  return merged;
}
