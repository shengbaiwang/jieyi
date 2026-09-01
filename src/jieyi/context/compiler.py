from __future__ import annotations

from jieyi.domain.models import Project, Segment, TermEntry, TermStatus
from jieyi.terminology import matching_terms, render_terminology_constraints


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
