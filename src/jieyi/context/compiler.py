from __future__ import annotations

from jieyi.domain.models import Project, Segment, TermEntry, TermStatus
from jieyi.terminology import matching_terms, render_terminology_constraints


def compile_neighbor_context(
    segment: Segment,
    neighbors: list[Segment],
    *,
    max_chars: int,
    include_translations: bool = False,
) -> str:
    """Render reference passages with a shared budget for both sides of the source."""
    sections: list[tuple[str, str, bool]] = []
    if segment.heading_path:
        sections.append(("# LOCATION", segment.heading_path, False))
    for neighbor in sorted(neighbors, key=lambda item: item.ordinal):
        if neighbor.document_id != segment.document_id or neighbor.ordinal == segment.ordinal:
            continue
        previous = neighbor.ordinal < segment.ordinal
        direction = "PREVIOUS" if previous else "FOLLOWING"
        sections.append(
            (f"# {direction} SOURCE (segment {neighbor.ordinal})", neighbor.source_text, previous)
        )
        translation = (
            neighbor.accepted_translation
            or neighbor.reviewed_translation
            or neighbor.edited_translation
            or neighbor.machine_translation
        )
        if include_translations and translation:
            sections.append(
                (f"# {direction} TRANSLATION (segment {neighbor.ordinal})", translation, previous)
            )
    if not sections:
        return ""

    prefix = (
        "# REFERENCE CONTEXT\n"
        "Use these passages to resolve references and maintain consistent wording. "
        "Do not translate, copy or repair neighboring passages; only the current source "
        "is the output target. Neighbor translations may contain errors; verify them "
        "against their source."
    )
    suffix = "\n\n# END REFERENCE CONTEXT"
    remaining = max_chars - len(prefix) - len(suffix) - sum(
        len(title) + 3 for title, _, _ in sections
    )
    if remaining < len(sections):
        return ""

    # Share space fairly, reclaiming unused space from short headings/passages.
    limits = [0] * len(sections)
    ordered = sorted(range(len(sections)), key=lambda index: len(sections[index][1]))
    for position, index in enumerate(ordered):
        limits[index] = min(len(sections[index][1]), remaining // (len(ordered) - position))
        remaining -= limits[index]
    rendered = [prefix]
    for (title, value, keep_tail), limit in zip(sections, limits, strict=True):
        if len(value) > limit:
            # Keep the text closest to the current segment when a neighbor is long.
            if limit <= 1:
                value = "…"[:limit]
            elif keep_tail:
                value = "…" + value[-(limit - 1):]
            else:
                value = value[:limit - 1] + "…"
        rendered.append(f"{title}\n{value}")
    return "\n\n".join(rendered) + suffix


class ContextCompiler:
    """Compile bounded, inspectable context instead of dumping an entire book."""

    def __init__(self, store):
        self.store = store

    def relevant_terms(self, project_id: str, source_text: str) -> list[TermEntry]:
        approved = [
            term for term in self.store.list_terms(project_id) if term.status is TermStatus.APPROVED
        ]
        return matching_terms(source_text, approved)

    def compile(
        self,
        project: Project,
        segment: Segment,
        *,
        neighbor_radius: int,
        max_chars: int,
        tm_enabled: bool = True,
        tm_threshold: float = 0.78,
        tm_max_results: int = 3,
    ) -> tuple[str, list[TermEntry]]:
        terms = self.relevant_terms(project.id, segment.source_text)
        neighbors = self.store.get_neighbors(
            segment.document_id, segment.ordinal, radius=neighbor_radius
        )

        sections: list[str] = [
            "# PROJECT",
            f"Source language: {project.source_lang}",
            f"Target language: {project.target_lang}",
            f"Domain: {project.domain}",
            f"Quote policy: {project.quote_policy}",
        ]
        if project.style_guide.strip():
            sections.extend(["", "# STYLE GUIDE", project.style_guide.strip()])

        if segment.heading_path:
            sections.extend(["", "# LOCATION", segment.heading_path])

        if terms:
            sections.extend(["", render_terminology_constraints(segment.source_text, terms)])

        if tm_enabled:
            tm_matches = self.store.search_translation_memory(
                project.id,
                segment.source_text,
                threshold=tm_threshold,
                limit=tm_max_results,
            )
            if tm_matches:
                sections.extend(["", "# TRANSLATION MEMORY (REFERENCE, NOT MANDATORY)"])
                for match in tm_matches:
                    percent = round(match.similarity * 100)
                    sections.append(f"- Match {percent}%")
                    sections.append(f"  Source: {match.source_text}")
                    sections.append(f"  Target: {match.target_text}")

        before = [item for item in neighbors if item.ordinal < segment.ordinal]
        after = [item for item in neighbors if item.ordinal > segment.ordinal]
        if before:
            sections.extend(["", "# PREVIOUS SOURCE"])
            for item in before:
                sections.append(item.source_text)
                known = item.accepted_translation or item.machine_translation
                if known:
                    sections.append(f"Previous translation: {known}")
        if after:
            sections.extend(["", "# FOLLOWING SOURCE"])
            sections.extend(item.source_text for item in after)

        context = "\n".join(sections)
        if len(context) > max_chars:
            context = context[: max(0, max_chars - 40)] + "\n[context truncated by budget]"
        return context, terms
