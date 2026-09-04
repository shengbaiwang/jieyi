from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from dataclasses import asdict, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from jieyi.domain.models import (
    CandidateStage,
    Document,
    Job,
    JobStatus,
    Project,
    QualityIssue,
    Segment,
    SegmentKind,
    SegmentStatus,
    TermEnforcement,
    TermEntry,
    TermStatus,
    TranslationMemoryMatch,
    TranslationRecipe,
    TranslationResult,
    new_id,
    utc_now,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_lang TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    domain TEXT NOT NULL,
    style_guide TEXT NOT NULL,
    quote_policy TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    source_format TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    stable_key TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    kind TEXT NOT NULL,
    source_text TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    machine_translation TEXT,
    edited_translation TEXT,
    reviewed_translation TEXT,
    accepted_translation TEXT,
    status TEXT NOT NULL,
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    segmentation_confidence REAL NOT NULL DEFAULT 1.0,
    segmentation_reason TEXT NOT NULL DEFAULT '',
    segmenter_version TEXT NOT NULL DEFAULT '',
    UNIQUE(document_id, stable_key),
    UNIQUE(document_id, ordinal)
);

CREATE TABLE IF NOT EXISTS terms (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    status TEXT NOT NULL,
    scope TEXT NOT NULL,
    domain TEXT NOT NULL,
    rationale TEXT NOT NULL,
    forbidden_targets_json TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    context_keywords_json TEXT NOT NULL DEFAULT '[]',
    sense TEXT NOT NULL DEFAULT '',
    disambiguation TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(project_id, source, scope, sense)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    recipe_json TEXT NOT NULL,
    status TEXT NOT NULL,
    next_ordinal INTEGER NOT NULL,
    total_cost_usd REAL NOT NULL,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    prompt_cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
    prompt_cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL,
    raw_response TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_batches (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    start_ordinal INTEGER NOT NULL,
    end_ordinal INTEGER NOT NULL,
    segment_count INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    reasoning_tokens INTEGER NOT NULL,
    prompt_cache_hit_tokens INTEGER NOT NULL,
    prompt_cache_miss_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    elapsed_seconds REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issues (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    severity TEXT NOT NULL,
    details_json TEXT NOT NULL,
    detector_version TEXT NOT NULL DEFAULT '1',
    target_hash TEXT NOT NULL DEFAULT '',
    resolved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    translation TEXT NOT NULL,
    rationale TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tm_entries (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_text TEXT NOT NULL,
    source_normalized TEXT NOT NULL,
    target_text TEXT NOT NULL,
    source_segment_id TEXT REFERENCES segments(id) ON DELETE SET NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, source_normalized)
);

CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS term_discovery_runs (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    config_json TEXT NOT NULL,
    coverage_json TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    reasoning_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS term_lexeme_candidates (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES term_discovery_runs(id) ON DELETE CASCADE,
    lexeme_key TEXT NOT NULL,
    canonical_form TEXT NOT NULL,
    forms_json TEXT NOT NULL,
    frequency INTEGER NOT NULL,
    segment_frequency INTEGER NOT NULL,
    risk_score REAL NOT NULL,
    rank INTEGER NOT NULL,
    candidate_type TEXT NOT NULL DEFAULT 'unclassified',
    boundary_confidence REAL NOT NULL DEFAULT 0,
    score_components_json TEXT NOT NULL,
    extraction_methods_json TEXT NOT NULL,
    UNIQUE(run_id, lexeme_key)
);

CREATE TABLE IF NOT EXISTS term_candidate_senses (
    id TEXT PRIMARY KEY,
    lexeme_id TEXT NOT NULL REFERENCES term_lexeme_candidates(id) ON DELETE CASCADE,
    sense_key TEXT NOT NULL,
    sense TEXT NOT NULL,
    concept_definition TEXT NOT NULL,
    proposed_target TEXT NOT NULL,
    rationale TEXT NOT NULL,
    disambiguation TEXT NOT NULL,
    confidence REAL NOT NULL,
    ai_recommended INTEGER,
    evidence_ids_json TEXT NOT NULL,
    proposer TEXT NOT NULL,
    status TEXT NOT NULL,
    context_keywords_json TEXT NOT NULL DEFAULT '[]',
    approved_term_id TEXT REFERENCES terms(id) ON DELETE SET NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS term_candidate_evidence (
    id TEXT PRIMARY KEY,
    lexeme_id TEXT NOT NULL REFERENCES term_lexeme_candidates(id) ON DELETE CASCADE,
    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    source_form TEXT NOT NULL,
    quote TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS epub_packages (
    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    original_epub BLOB NOT NULL,
    package_path TEXT NOT NULL,
    package_version TEXT NOT NULL,
    nav_path TEXT,
    cover_path TEXT,
    rendition_layout TEXT NOT NULL,
    page_progression_direction TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    attached_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS epub_resources (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    properties TEXT NOT NULL,
    data BLOB NOT NULL,
    sha256 TEXT NOT NULL,
    PRIMARY KEY (document_id, path)
);

CREATE TABLE IF NOT EXISTS epub_spine (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    spine_index INTEGER NOT NULL,
    idref TEXT NOT NULL,
    path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    properties TEXT NOT NULL,
    linear INTEGER NOT NULL,
    fixed_layout INTEGER NOT NULL,
    PRIMARY KEY (document_id, spine_index)
);

CREATE TABLE IF NOT EXISTS epub_atoms (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    atom_id TEXT NOT NULL,
    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    spine_index INTEGER NOT NULL,
    spine_path TEXT NOT NULL,
    dom_path TEXT NOT NULL,
    semantic_path TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    source_markup TEXT NOT NULL,
    node_refs_json TEXT NOT NULL,
    PRIMARY KEY (document_id, atom_id)
);

CREATE TABLE IF NOT EXISTS epub_text_nodes (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    atom_id TEXT NOT NULL,
    spine_path TEXT NOT NULL,
    dom_path TEXT NOT NULL,
    slot TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    PRIMARY KEY (document_id, node_id)
);

CREATE TABLE IF NOT EXISTS epub_atom_translations (
    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    atom_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    translation_text TEXT NOT NULL,
    translation_markup TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (segment_id, atom_id, stage)
);

CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_document_ordinal ON segments(document_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_terms_project ON terms(project_id);
CREATE INDEX IF NOT EXISTS idx_candidates_segment ON candidates(segment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_job_batches_job ON job_batches(job_id, created_at);
CREATE INDEX IF NOT EXISTS idx_issues_segment ON issues(segment_id, resolved);
CREATE INDEX IF NOT EXISTS idx_tm_project_source ON tm_entries(project_id, source_normalized);
CREATE INDEX IF NOT EXISTS idx_documents_source_hash ON documents(source_hash);
CREATE INDEX IF NOT EXISTS idx_term_runs_document ON term_discovery_runs(document_id, created_at);
CREATE INDEX IF NOT EXISTS idx_term_lexemes_run ON term_lexeme_candidates(run_id, rank);
CREATE INDEX IF NOT EXISTS idx_term_senses_status ON term_candidate_senses(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_term_evidence_lexeme ON term_candidate_evidence(lexeme_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_epub_resources_document ON epub_resources(document_id);
CREATE INDEX IF NOT EXISTS idx_epub_atoms_segment ON epub_atoms(segment_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_epub_atoms_spine ON epub_atoms(document_id, spine_index, ordinal);
"""


class NotFoundError(LookupError):
    pass


class _ClosingConnection(sqlite3.Connection):
    """Make ``with connection`` commit/rollback and then release the handle."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class SQLiteStore:
    """Durable store with explicit records for candidates, decisions, and issues."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "sha256",
            1,
            lambda value: hashlib.sha256(str(value or "").encode("utf-8")).hexdigest(),
            deterministic=True,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def migrate(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            segment_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(segments)").fetchall()
            }
            if "edited_translation" not in segment_columns:
                connection.execute("ALTER TABLE segments ADD COLUMN edited_translation TEXT")
            if "reviewed_translation" not in segment_columns:
                connection.execute("ALTER TABLE segments ADD COLUMN reviewed_translation TEXT")
            if "source_refs_json" not in segment_columns:
                connection.execute(
                    "ALTER TABLE segments ADD COLUMN source_refs_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "segmentation_confidence" not in segment_columns:
                connection.execute(
                    "ALTER TABLE segments ADD COLUMN segmentation_confidence "
                    "REAL NOT NULL DEFAULT 1.0"
                )
            if "segmentation_reason" not in segment_columns:
                connection.execute(
                    "ALTER TABLE segments ADD COLUMN segmentation_reason TEXT NOT NULL DEFAULT ''"
                )
            if "segmenter_version" not in segment_columns:
                connection.execute(
                    "ALTER TABLE segments ADD COLUMN segmenter_version TEXT NOT NULL DEFAULT ''"
                )
            term_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(terms)").fetchall()
            }
            required_term_columns = {
                "aliases_json",
                "context_keywords_json",
                "sense",
                "disambiguation",
            }
            if not required_term_columns.issubset(term_columns):
                connection.execute("ALTER TABLE terms RENAME TO terms_legacy")
                connection.execute(
                    """CREATE TABLE terms (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    status TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    forbidden_targets_json TEXT NOT NULL,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    context_keywords_json TEXT NOT NULL DEFAULT '[]',
                    sense TEXT NOT NULL DEFAULT '',
                    disambiguation TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(project_id, source, scope, sense)
                    )"""
                )
                aliases_value = "aliases_json" if "aliases_json" in term_columns else "'[]'"
                context_value = (
                    "context_keywords_json" if "context_keywords_json" in term_columns else "'[]'"
                )
                sense_value = "sense" if "sense" in term_columns else "''"
                disambiguation_value = (
                    "disambiguation" if "disambiguation" in term_columns else "''"
                )
                connection.execute(
                    f"""INSERT INTO terms
                    (id, project_id, source, target, status, scope, domain, rationale,
                     forbidden_targets_json, aliases_json, context_keywords_json, sense,
                     disambiguation, created_at)
                    SELECT id, project_id, source, target, status, scope, domain, rationale,
                    forbidden_targets_json,
                    {aliases_value},
                    {context_value},
                    {sense_value},
                    {disambiguation_value},
                    created_at FROM terms_legacy"""
                )
                connection.execute("DROP TABLE terms_legacy")
                connection.execute("CREATE INDEX idx_terms_project ON terms(project_id)")
            if "enforcement" not in {
                row["name"] for row in connection.execute("PRAGMA table_info(terms)")
            }:
                connection.execute(
                    "ALTER TABLE terms ADD COLUMN enforcement TEXT NOT NULL DEFAULT 'auto'"
                )
            from jieyi.quality.terminology_review import _SCHEMA as review_schema

            connection.executescript(review_schema)
            sense_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(term_candidate_senses)")
            }
            if "context_keywords_json" not in sense_columns:
                connection.execute(
                    "ALTER TABLE term_candidate_senses ADD COLUMN "
                    "context_keywords_json TEXT NOT NULL DEFAULT '[]'"
                )
                connection.execute(
                    """UPDATE term_candidate_senses SET context_keywords_json = COALESCE(
                    (SELECT context_keywords_json FROM terms WHERE id = approved_term_id), '[]')"""
                )
            issue_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(issues)").fetchall()
            }
            if "detector_version" not in issue_columns:
                connection.execute(
                    "ALTER TABLE issues ADD COLUMN detector_version TEXT NOT NULL DEFAULT '1'"
                )
            if "target_hash" not in issue_columns:
                connection.execute(
                    "ALTER TABLE issues ADD COLUMN target_hash TEXT NOT NULL DEFAULT ''"
                )
            term_candidate_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(term_lexeme_candidates)"
                ).fetchall()
            }
            if "candidate_type" not in term_candidate_columns:
                connection.execute(
                    "ALTER TABLE term_lexeme_candidates ADD COLUMN "
                    "candidate_type TEXT NOT NULL DEFAULT 'unclassified'"
                )
            if "boundary_confidence" not in term_candidate_columns:
                connection.execute(
                    "ALTER TABLE term_lexeme_candidates ADD COLUMN "
                    "boundary_confidence REAL NOT NULL DEFAULT 0"
                )
            candidate_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(candidates)").fetchall()
            }
            for name in ("reasoning_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                if name not in candidate_columns:
                    connection.execute(
                        f"ALTER TABLE candidates ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                    )
            usage_rows = connection.execute(
                "SELECT id, raw_response FROM candidates WHERE raw_response IS NOT NULL"
            ).fetchall()
            usage_updates = []
            for row in usage_rows:
                try:
                    usage = (json.loads(row["raw_response"]) or {}).get("usage") or {}
                    details = usage.get("completion_tokens_details") or {}
                    usage_updates.append(
                        (
                            int(details.get("reasoning_tokens") or 0),
                            int(usage.get("prompt_cache_hit_tokens") or 0),
                            int(usage.get("prompt_cache_miss_tokens") or 0),
                            row["id"],
                        )
                    )
                except (TypeError, ValueError, AttributeError):
                    continue
            if usage_updates:
                connection.executemany(
                    """UPDATE candidates SET reasoning_tokens = ?,
                    prompt_cache_hit_tokens = ?, prompt_cache_miss_tokens = ? WHERE id = ?""",
                    usage_updates,
                )

    def create_project(self, project: Project) -> Project:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO projects
                (id, name, source_lang, target_lang, domain, style_guide, quote_policy, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project.id,
                    project.name,
                    project.source_lang,
                    project.target_lang,
                    project.domain,
                    project.style_guide,
                    project.quote_policy,
                    project.created_at,
                ),
            )
            self._audit(connection, "project", project.id, "created", {"name": project.name})
        return project

    def get_project(self, project_id: str) -> Project:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Project not found: {project_id}")
        return self._project(row)

    def update_project_style(self, project_id: str, style_guide: str) -> Project:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE projects SET style_guide = ? WHERE id = ?",
                (style_guide, project_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"Project not found: {project_id}")
            self._audit(
                connection,
                "project",
                project_id,
                "style_updated",
                {"style_guide": style_guide},
            )
        return self.get_project(project_id)

    def update_project_languages(
        self, project_id: str, source_lang: str, target_lang: str
    ) -> Project:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE projects SET source_lang = ?, target_lang = ? WHERE id = ?",
                (source_lang, target_lang, project_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"Project not found: {project_id}")
            self._audit(
                connection,
                "project",
                project_id,
                "languages_updated",
                {"source_lang": source_lang, "target_lang": target_lang},
            )
        return self.get_project(project_id)

    def find_document_by_source_hash(self, project_id: str, source_hash: str) -> Document | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM documents WHERE project_id = ? AND source_hash = ?
                ORDER BY created_at LIMIT 1""",
                (project_id, source_hash),
            ).fetchone()
        return self._document(row) if row is not None else None

    def attach_epub_archive(self, document_id: str, archive) -> None:
        """Attach or refresh byte-perfect EPUB resources without touching translation records."""
        document = self.get_document(document_id)
        if document.source_format != "epub":
            raise ValueError("EPUB resources can only be attached to EPUB documents")
        if document.source_hash != archive.source_hash:
            raise ValueError("EPUB SHA-256 does not match the imported document")

        segments = self.list_segments(document_id)
        ref_to_segment = {ref: segment.id for segment in segments for ref in segment.source_refs}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for table in (
                "epub_text_nodes",
                "epub_atoms",
                "epub_spine",
                "epub_resources",
                "epub_packages",
            ):
                connection.execute(
                    f"DELETE FROM {table} WHERE document_id = ?",
                    (document_id,),
                )
            connection.execute(
                """INSERT INTO epub_packages
                (document_id, original_epub, package_path, package_version, nav_path,
                 cover_path, rendition_layout, page_progression_direction,
                 metadata_json, source_hash, attached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    archive.original_data,
                    archive.package_path,
                    archive.package_version,
                    archive.nav_path,
                    archive.cover_path,
                    archive.rendition_layout,
                    archive.page_progression_direction,
                    json.dumps(archive.metadata, ensure_ascii=False),
                    archive.source_hash,
                    utc_now(),
                ),
            )
            connection.executemany(
                """INSERT INTO epub_resources
                (document_id, path, media_type, properties, data, sha256)
                VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        document_id,
                        item.path,
                        item.media_type,
                        item.properties,
                        item.data,
                        item.sha256,
                    )
                    for item in archive.resources
                ],
            )
            connection.executemany(
                """INSERT INTO epub_spine
                (document_id, spine_index, idref, path, media_type, properties,
                 linear, fixed_layout)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        document_id,
                        item.index,
                        item.idref,
                        item.path,
                        item.media_type,
                        item.properties,
                        int(item.linear),
                        int(item.fixed_layout),
                    )
                    for item in archive.spine
                ],
            )
            atom_rows = [
                (
                    document_id,
                    item.atom_id,
                    ref_to_segment[item.atom_id],
                    item.spine_index,
                    item.spine_path,
                    item.dom_path,
                    item.semantic_path,
                    item.ordinal,
                    item.source_text,
                    item.source_markup,
                    json.dumps(item.node_refs, ensure_ascii=False),
                )
                for item in archive.atoms
                if item.atom_id in ref_to_segment
            ]
            connection.executemany(
                """INSERT INTO epub_atoms
                (document_id, atom_id, segment_id, spine_index, spine_path,
                 dom_path, semantic_path, ordinal, source_text, source_markup,
                 node_refs_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                atom_rows,
            )
            persisted_atoms = {row[1] for row in atom_rows}
            connection.executemany(
                """INSERT INTO epub_text_nodes
                (document_id, node_id, atom_id, spine_path, dom_path, slot,
                 ordinal, source_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        document_id,
                        item.node_id,
                        item.atom_id,
                        item.spine_path,
                        item.dom_path,
                        item.slot,
                        item.ordinal,
                        item.source_text,
                    )
                    for item in archive.text_nodes
                    if item.atom_id in persisted_atoms
                ],
            )
            self._audit(
                connection,
                "document",
                document_id,
                "epub_resources_attached",
                {
                    "source_hash": archive.source_hash,
                    "resources": len(archive.resources),
                    "spine_items": len(archive.spine),
                    "atoms": len(atom_rows),
                },
            )

    def has_epub_archive(self, document_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM epub_packages WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return row is not None

    def get_epub_package(self, document_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT document_id, package_path, package_version, nav_path,
                cover_path, rendition_layout, page_progression_direction,
                metadata_json, source_hash, attached_at, length(original_epub) AS byte_length
                FROM epub_packages WHERE document_id = ?""",
                (document_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"EPUB resources are not attached: {document_id}")
        value = dict(row)
        value["metadata"] = dict(json.loads(value.pop("metadata_json")))
        return value

    def get_original_epub(self, document_id: str) -> bytes:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT original_epub FROM epub_packages WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"EPUB resources are not attached: {document_id}")
        return bytes(row["original_epub"])

    def list_epub_spine(self, document_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM epub_spine WHERE document_id = ?
                ORDER BY spine_index""",
                (document_id,),
            ).fetchall()
        return [
            dict(row)
            | {
                "linear": bool(row["linear"]),
                "fixed_layout": bool(row["fixed_layout"]),
            }
            for row in rows
        ]

    def list_epub_segment_locations(self, document_id: str) -> list[dict[str, int]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT s.ordinal AS segment_ordinal, a.spine_index
                FROM epub_atoms a JOIN segments s ON s.id = a.segment_id
                WHERE a.document_id = ? ORDER BY a.ordinal""",
                (document_id,),
            ).fetchall()
        result: list[dict[str, int]] = []
        seen: set[int] = set()
        for row in rows:
            ordinal = int(row["segment_ordinal"])
            if ordinal in seen:
                continue
            seen.add(ordinal)
            result.append({"segment_ordinal": ordinal, "spine_index": int(row["spine_index"])})
        return result

    def get_epub_resource(self, document_id: str, path: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM epub_resources WHERE document_id = ? AND path = ?",
                (document_id, path),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"EPUB resource not found: {path}")
        return dict(row)

    def list_epub_atoms_for_segment(self, segment_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM epub_atoms WHERE segment_id = ? ORDER BY ordinal""",
                (segment_id,),
            ).fetchall()
        return [dict(row) | {"node_refs": json.loads(row["node_refs_json"])} for row in rows]

    def list_epub_locations_for_spine(
        self, document_id: str, spine_index: int
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT a.dom_path, s.ordinal AS segment_ordinal
                FROM epub_atoms a JOIN segments s ON s.id = a.segment_id
                WHERE a.document_id = ? AND a.spine_index = ?
                ORDER BY a.ordinal""",
                (document_id, spine_index),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_epub_atoms_for_spine(self, document_id: str, spine_index: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT a.*, s.ordinal AS segment_ordinal,
                s.machine_translation, s.edited_translation,
                s.reviewed_translation, s.accepted_translation, s.status
                FROM epub_atoms a JOIN segments s ON s.id = a.segment_id
                WHERE a.document_id = ? AND a.spine_index = ?
                ORDER BY a.ordinal""",
                (document_id, spine_index),
            ).fetchall()
            translations = connection.execute(
                """SELECT DISTINCT t.* FROM epub_atom_translations t
                JOIN epub_atoms a
                  ON a.segment_id = t.segment_id AND a.atom_id = t.atom_id
                WHERE a.document_id = ? AND a.spine_index = ?""",
                (document_id, spine_index),
            ).fetchall()
        by_atom: dict[str, dict[str, sqlite3.Row]] = {}
        for row in translations:
            by_atom.setdefault(row["atom_id"], {})[row["stage"]] = row
        result: list[dict[str, Any]] = []
        first_by_segment: set[str] = set()
        for row in rows:
            value = dict(row)
            stage_rows = by_atom.get(row["atom_id"], {})
            if row["accepted_translation"]:
                allowed_stages = ("accepted",)
            elif row["reviewed_translation"]:
                allowed_stages = ("reviewed", "review", "repair")
            elif row["edited_translation"]:
                allowed_stages = ("edited",)
            else:
                allowed_stages = ("machine", "review", "draft", "repair")
            chosen = next(
                (stage_rows[stage] for stage in allowed_stages if stage in stage_rows),
                None,
            )
            if chosen is not None:
                value["translation_text"] = chosen["translation_text"]
                value["translation_markup"] = chosen["translation_markup"]
            elif row["segment_id"] not in first_by_segment:
                fallback = (
                    row["accepted_translation"]
                    or row["reviewed_translation"]
                    or row["edited_translation"]
                    or row["machine_translation"]
                    or ""
                )
                value["translation_text"] = fallback
                value["translation_markup"] = fallback
            else:
                value["translation_text"] = ""
                value["translation_markup"] = ""
            first_by_segment.add(row["segment_id"])
            value["node_refs"] = json.loads(row["node_refs_json"])
            result.append(value)
        return result

    def list_epub_text_nodes_for_spine(
        self, document_id: str, spine_index: int
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT n.* FROM epub_text_nodes n
                JOIN epub_atoms a
                  ON a.document_id = n.document_id AND a.atom_id = n.atom_id
                WHERE n.document_id = ? AND a.spine_index = ?
                ORDER BY n.ordinal""",
                (document_id, spine_index),
            ).fetchall()
        return [dict(row) for row in rows]

    def epub_translation_source(self, segment_id: str) -> str | None:
        from jieyi.ingestion.epub_roundtrip import build_structured_source

        atoms = self.list_epub_atoms_for_segment(segment_id)
        return build_structured_source(atoms) if atoms else None

    def epub_structured_translation(self, segment_id: str) -> str | None:
        from html import escape

        atoms = self.list_epub_atoms_for_segment(segment_id)
        if not atoms:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM epub_atom_translations WHERE segment_id = ?""",
                (segment_id,),
            ).fetchall()
        by_atom: dict[str, dict[str, sqlite3.Row]] = {}
        for row in rows:
            by_atom.setdefault(row["atom_id"], {})[row["stage"]] = row
        priority = ("accepted", "edited", "reviewed", "machine", "review", "draft", "repair")
        pieces: list[str] = []
        for atom in atoms:
            stage_rows = by_atom.get(atom["atom_id"], {})
            chosen = next(
                (stage_rows[stage] for stage in priority if stage in stage_rows),
                None,
            )
            if chosen is None:
                return None
            pieces.append(
                f'<jy-atom data-jy-id="{escape(atom["atom_id"], quote=True)}">'
                f"{chosen['translation_markup']}</jy-atom>"
            )
        return "".join(pieces)

    def capture_epub_translation(self, segment_id: str, value: str, stage: str) -> str:
        from jieyi.ingestion.epub_roundtrip import parse_structured_translation

        atoms = self.list_epub_atoms_for_segment(segment_id)
        if not atoms:
            return value
        expected = tuple(item["atom_id"] for item in atoms)
        plain, translations = parse_structured_translation(value, expected)
        with self._connect() as connection:
            connection.executemany(
                """INSERT INTO epub_atom_translations
                (segment_id, atom_id, stage, translation_text,
                 translation_markup, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id, atom_id, stage) DO UPDATE SET
                translation_text = excluded.translation_text,
                translation_markup = excluded.translation_markup,
                updated_at = excluded.updated_at""",
                [
                    (
                        segment_id,
                        atom_id,
                        stage,
                        translation[0],
                        translation[1],
                        utc_now(),
                    )
                    for atom_id, translation in translations.items()
                ],
            )
        return plain

    def list_epub_mappings(self, document_id: str) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as connection:
            atoms = connection.execute(
                """SELECT atom_id, segment_id, spine_index, spine_path, dom_path,
                semantic_path, ordinal, source_text, node_refs_json
                FROM epub_atoms WHERE document_id = ? ORDER BY ordinal""",
                (document_id,),
            ).fetchall()
            nodes = connection.execute(
                """SELECT node_id, atom_id, spine_path, dom_path, slot, ordinal, source_text
                FROM epub_text_nodes WHERE document_id = ? ORDER BY ordinal""",
                (document_id,),
            ).fetchall()
        return {
            "atoms": [
                dict(row) | {"node_refs": json.loads(row["node_refs_json"])} for row in atoms
            ],
            "text_nodes": [dict(row) for row in nodes],
        }

    def list_projects(self) -> list[Project]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [self._project(row) for row in rows]

    def get_project_for_document(self, document_id: str) -> Project:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT p.* FROM projects p
                JOIN documents d ON d.project_id = p.id WHERE d.id = ?""",
                (document_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Document not found: {document_id}")
        return self._project(row)

    def create_document(self, document: Document, segments: list[Segment]) -> Document:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO documents
                (id, project_id, title, source_format, source_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    document.id,
                    document.project_id,
                    document.title,
                    document.source_format,
                    document.source_hash,
                    document.created_at,
                ),
            )
            connection.executemany(
                """INSERT INTO segments
                (id, document_id, stable_key, ordinal, kind, source_text, heading_path,
                 machine_translation, edited_translation, reviewed_translation,
                 accepted_translation, status, source_refs_json, segmentation_confidence,
                 segmentation_reason, segmenter_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        item.id,
                        item.document_id,
                        item.stable_key,
                        item.ordinal,
                        item.kind.value,
                        item.source_text,
                        item.heading_path,
                        item.machine_translation,
                        item.edited_translation,
                        item.reviewed_translation,
                        item.accepted_translation,
                        item.status.value,
                        json.dumps(item.source_refs, ensure_ascii=False),
                        item.segmentation_confidence,
                        item.segmentation_reason,
                        item.segmenter_version,
                    )
                    for item in segments
                ],
            )
            self._audit(
                connection,
                "document",
                document.id,
                "imported",
                {"segments": len(segments), "source_hash": document.source_hash},
            )
        return document

    def get_document(self, document_id: str) -> Document:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Document not found: {document_id}")
        return self._document(row)

    def list_documents(self, project_id: str) -> list[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [self._document(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        document = self.get_document(document_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._audit(
                connection,
                "document",
                document_id,
                "deleted",
                {"title": document.title, "project_id": document.project_id},
            )
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def list_segments(self, document_id: str) -> list[Segment]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM segments WHERE document_id = ? ORDER BY ordinal", (document_id,)
            ).fetchall()
        return [self._segment(row) for row in rows]

    def list_segments_page(
        self, document_id: str, *, offset: int = 0, limit: int = 100
    ) -> tuple[int, list[Segment]]:
        with self._connect() as connection:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM segments WHERE document_id = ?", (document_id,)
                ).fetchone()[0]
            )
            rows = connection.execute(
                """SELECT * FROM segments WHERE document_id = ?
                ORDER BY ordinal LIMIT ? OFFSET ?""",
                (document_id, limit, offset),
            ).fetchall()
        return total, [self._segment(row) for row in rows]

    def search_segments(self, document_id: str, query: str, *, limit: int = 30) -> list[Segment]:
        pattern = f"%{query.strip()}%"
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM segments WHERE document_id = ?
                AND (source_text LIKE ? OR COALESCE(edited_translation, '') LIKE ?
                OR COALESCE(reviewed_translation, '') LIKE ?
                OR COALESCE(machine_translation, '') LIKE ?
                OR COALESCE(accepted_translation, '') LIKE ?)
                ORDER BY ordinal LIMIT ?""",
                (document_id, pattern, pattern, pattern, pattern, pattern, limit),
            ).fetchall()
        return [self._segment(row) for row in rows]

    def list_jobs(self, document_id: str) -> list[Job]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE document_id = ? ORDER BY created_at DESC",
                (document_id,),
            ).fetchall()
        return [self._job(row) for row in rows]

    def get_segment(self, segment_id: str) -> Segment:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM segments WHERE id = ?", (segment_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Segment not found: {segment_id}")
        return self._segment(row)

    def get_segment_source_blocks(self, segment_id: str) -> list[str]:
        segment = self.get_segment(segment_id)
        saved_blocks = [item.strip() for item in re.split(r"\n\s*\n", segment.source_text)]
        if len(saved_blocks) > 1:
            return saved_blocks
        if not segment.source_refs:
            return [segment.source_text]
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT atom_id, source_text FROM epub_atoms
                WHERE segment_id = ? ORDER BY ordinal""",
                (segment_id,),
            ).fetchall()
        by_id = {str(row["atom_id"]): str(row["source_text"]) for row in rows}
        blocks = [by_id[ref] for ref in segment.source_refs if ref in by_id]
        if not blocks:
            return [segment.source_text]
        normalized_source = re.sub(r"\s+", " ", segment.source_text).strip()
        normalized_blocks = re.sub(r"\s+", " ", " ".join(blocks)).strip()
        return blocks if normalized_source == normalized_blocks else [segment.source_text]

    def update_segment_source(
        self,
        segment_id: str,
        source_text: str,
        *,
        preserve_translation_for_review: bool = False,
    ) -> Segment:
        value = source_text.strip()
        if not value:
            raise ValueError("原文不能为空")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM segments WHERE id = ?", (segment_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Segment not found: {segment_id}")
            if value == str(row["source_text"]):
                return self._segment(row)
            translation = self._effective_translation(row)
            if translation and not preserve_translation_for_review:
                raise ValueError("本段已有译文；修改原文前必须确认将译文标记为待复核")
            self._clear_segment_derivatives(connection, [segment_id])
            connection.execute(
                """UPDATE segments SET source_text = ?, machine_translation = NULL,
                edited_translation = ?, reviewed_translation = NULL,
                accepted_translation = NULL, status = ? WHERE id = ?""",
                (
                    value,
                    translation or None,
                    SegmentStatus.MACHINE_TRANSLATED.value
                    if translation
                    else SegmentStatus.SOURCE.value,
                    segment_id,
                ),
            )
            self._audit(
                connection,
                "segment",
                segment_id,
                "source_updated",
                {
                    "characters": len(value),
                    "paragraphs": len(re.split(r"\n\s*\n", value)),
                    "translation_preserved_for_review": bool(translation),
                },
            )
        return self.get_segment(segment_id)

    def set_segment_heading(self, segment_id: str, *, heading: bool) -> Segment:
        """Promote an existing segment to a manual TOC heading, or restore it to body text."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM segments WHERE id = ?", (segment_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Segment not found: {segment_id}")
            self._ensure_no_active_structure_job(connection, str(row["document_id"]))

            old_kind = str(row["kind"])
            new_kind = (
                SegmentKind.HEADING.value if heading else SegmentKind.PARAGRAPH.value
            )
            if old_kind == new_kind:
                return self._segment(row)

            reason = (
                "manual_heading"
                if heading
                else "manual_heading_reverted"
            )
            connection.execute(
                """UPDATE segments SET kind = ?, segmentation_confidence = 1.0,
                segmentation_reason = ?, segmenter_version = 'manual-v2'
                WHERE id = ?""",
                (new_kind, reason, segment_id),
            )
            self._audit(
                connection,
                "segment",
                segment_id,
                "kind_updated",
                {"from": old_kind, "to": new_kind},
            )
        return self.get_segment(segment_id)

    @staticmethod
    def _effective_translation(row: sqlite3.Row) -> str:
        return str(
            row["accepted_translation"]
            or row["reviewed_translation"]
            or row["edited_translation"]
            or row["machine_translation"]
            or ""
        ).strip()

    @staticmethod
    def _source_blocks_for_row(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> list[str]:
        saved_blocks = [
            item.strip() for item in re.split(r"\n\s*\n", str(row["source_text"]))
        ]
        if len(saved_blocks) > 1:
            return saved_blocks
        refs = tuple(json.loads(row["source_refs_json"] or "[]"))
        if not refs:
            return [str(row["source_text"])]
        atom_rows = connection.execute(
            """SELECT atom_id, source_text FROM epub_atoms
            WHERE segment_id = ? ORDER BY ordinal""",
            (row["id"],),
        ).fetchall()
        by_id = {str(item["atom_id"]): str(item["source_text"]) for item in atom_rows}
        blocks = [by_id[ref] for ref in refs if ref in by_id]
        if not blocks:
            return [str(row["source_text"])]
        normalized_source = re.sub(r"\s+", " ", str(row["source_text"])).strip()
        normalized_blocks = re.sub(r"\s+", " ", " ".join(blocks)).strip()
        return blocks if normalized_source == normalized_blocks else [str(row["source_text"])]

    @staticmethod
    def _ensure_no_active_structure_job(
        connection: sqlite3.Connection, document_id: str
    ) -> None:
        row = connection.execute(
            """SELECT status FROM jobs WHERE document_id = ?
            AND status IN ('pending', 'running', 'paused', 'failed')
            ORDER BY created_at DESC LIMIT 1""",
            (document_id,),
        ).fetchone()
        if row is not None:
            raise ValueError("请先结束或取消未完成的翻译任务，再调整段落结构")

    @staticmethod
    def _clear_segment_derivatives(
        connection: sqlite3.Connection, segment_ids: list[str]
    ) -> None:
        if not segment_ids:
            return
        placeholders = ",".join("?" for _ in segment_ids)
        for table in (
            "candidates",
            "issues",
            "decisions",
            "term_candidate_evidence",
            "epub_atom_translations",
        ):
            connection.execute(
                f"DELETE FROM {table} WHERE segment_id IN ({placeholders})",
                segment_ids,
            )
        connection.execute(
            f"DELETE FROM tm_entries WHERE source_segment_id IN ({placeholders})",
            segment_ids,
        )

    @staticmethod
    def _renumber_segments(
        connection: sqlite3.Connection, document_id: str, ordered_ids: list[str]
    ) -> None:
        for index, item_id in enumerate(ordered_ids):
            connection.execute(
                "UPDATE segments SET ordinal = ? WHERE id = ?",
                (-(index + 1), item_id),
            )
        for index, item_id in enumerate(ordered_ids):
            connection.execute(
                "UPDATE segments SET ordinal = ? WHERE id = ? AND document_id = ?",
                (index, item_id, document_id),
            )

    def split_segment(
        self,
        segment_id: str,
        *,
        source_text: str,
        selection_start: int,
        selection_end: int,
        reset_translation: bool = False,
        preserve_translation: bool = False,
        selected_as_heading: bool = False,
    ) -> dict[str, Any]:
        """Turn a selected source range into its own translation unit atomically."""
        if reset_translation and preserve_translation:
            raise ValueError("拆分时不能同时选择清空译文和保留译文")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT s.*, d.source_format FROM segments s
                JOIN documents d ON d.id = s.document_id WHERE s.id = ?""",
                (segment_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Segment not found: {segment_id}")
            if row["kind"] == SegmentKind.HEADING.value:
                raise ValueError("标题不能拆为正文段落")
            document_id = str(row["document_id"])
            self._ensure_no_active_structure_job(connection, document_id)

            blocks = self._source_blocks_for_row(connection, row)
            displayed = "\n\n".join(blocks)
            if source_text != displayed:
                raise ValueError("选中后原文发生了变化，请重新载入本段后再试")
            if not (0 <= selection_start < selection_end <= len(displayed)):
                raise ValueError("请先在原文中选中一段非空文字")

            selected_raw = displayed[selection_start:selection_end]
            selection_start += len(selected_raw) - len(selected_raw.lstrip())
            selection_end -= len(selected_raw) - len(selected_raw.rstrip())
            if selection_start >= selection_end:
                raise ValueError("所选内容不能只有空白字符")

            raw_ranges = (
                (0, selection_start),
                (selection_start, selection_end),
                (selection_end, len(displayed)),
            )
            parts: list[str] = []
            part_ranges: list[tuple[int, int]] = []
            selected_part_index = -1
            for raw_index, (start, end) in enumerate(raw_ranges):
                raw = displayed[start:end]
                trimmed_start = start + len(raw) - len(raw.lstrip())
                trimmed_end = end - (len(raw) - len(raw.rstrip()))
                if trimmed_start >= trimmed_end:
                    continue
                if raw_index == 1:
                    selected_part_index = len(parts)
                parts.append(displayed[trimmed_start:trimmed_end])
                part_ranges.append((trimmed_start, trimmed_end))
            if len(parts) < 2 or selected_part_index < 0:
                raise ValueError("请选择本段中的一部分，而不是整段")

            refs = tuple(json.loads(row["source_refs_json"] or "[]"))
            part_refs: list[tuple[str, ...]]
            if row["source_format"] == "epub":
                if not refs:
                    raise ValueError("该 EPUB 段落没有可安全回写的原始文本位置")
                atom_rows = connection.execute(
                    """SELECT * FROM epub_atoms WHERE document_id = ?
                    AND segment_id = ? ORDER BY ordinal""",
                    (document_id, segment_id),
                ).fetchall()
                atoms_by_id = {str(atom["atom_id"]): atom for atom in atom_rows}
                if any(ref not in atoms_by_id for ref in refs):
                    raise ValueError("该 EPUB 段落的原始文本映射不完整，请重新导入后再试")
                ordered_atoms = [atoms_by_id[ref] for ref in refs]

                atom_positions: list[tuple[sqlite3.Row, int, int]] = []
                cursor = 0
                for atom in ordered_atoms:
                    atom_text = str(atom["source_text"])
                    start = displayed.find(atom_text, cursor)
                    if start < 0:
                        raise ValueError("该 EPUB 段落的显示文本与原书位置不一致，无法安全拆分")
                    end = start + len(atom_text)
                    atom_positions.append((atom, start, end))
                    cursor = end

                from html import escape

                all_atom_ids = [
                    str(item["atom_id"])
                    for item in connection.execute(
                        "SELECT atom_id FROM epub_atoms WHERE document_id = ? ORDER BY ordinal",
                        (document_id,),
                    ).fetchall()
                ]
                replacements: dict[str, list[str]] = {}
                refs_by_part: list[list[str]] = [[] for _ in parts]
                for atom, atom_start, atom_end in atom_positions:
                    atom_id = str(atom["atom_id"])
                    atom_text = str(atom["source_text"])
                    cuts = {0, len(atom_text)}
                    for boundary in (selection_start, selection_end):
                        if atom_start < boundary < atom_end:
                            cuts.add(boundary - atom_start)
                    ordered_cuts = sorted(cuts)
                    chunks = [
                        (ordered_cuts[index], ordered_cuts[index + 1])
                        for index in range(len(ordered_cuts) - 1)
                        if ordered_cuts[index] < ordered_cuts[index + 1]
                    ]
                    chunk_ids = [atom_id] + [
                        new_id("atom") for _ in range(max(0, len(chunks) - 1))
                    ]
                    replacements[atom_id] = chunk_ids
                    if len(chunks) > 1:
                        first_start, first_end = chunks[0]
                        first_text = atom_text[first_start:first_end].strip()
                        connection.execute(
                            """UPDATE epub_atoms SET source_text = ?, source_markup = ?
                            WHERE document_id = ? AND atom_id = ?""",
                            (first_text, escape(first_text), document_id, atom_id),
                        )
                        for child_id, (chunk_start, chunk_end) in zip(
                            chunk_ids[1:], chunks[1:], strict=True
                        ):
                            chunk_text = atom_text[chunk_start:chunk_end].strip()
                            connection.execute(
                                """INSERT INTO epub_atoms
                                (document_id, atom_id, segment_id, spine_index, spine_path,
                                 dom_path, semantic_path, ordinal, source_text, source_markup,
                                 node_refs_json)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    document_id,
                                    child_id,
                                    segment_id,
                                    atom["spine_index"],
                                    atom["spine_path"],
                                    atom["dom_path"],
                                    atom["semantic_path"],
                                    -(len(all_atom_ids) + len(chunk_ids)),
                                    chunk_text,
                                    escape(chunk_text),
                                    atom["node_refs_json"],
                                ),
                            )
                    for child_id, (chunk_start, chunk_end) in zip(
                        chunk_ids, chunks, strict=True
                    ):
                        global_start = atom_start + chunk_start
                        global_end = atom_start + chunk_end
                        overlaps = [
                            max(0, min(global_end, part_end) - max(global_start, part_start))
                            for part_start, part_end in part_ranges
                        ]
                        best_part = max(range(len(parts)), key=overlaps.__getitem__)
                        if overlaps[best_part] <= 0:
                            raise ValueError("EPUB 文本边界包含无法归属的内容，拆分已取消")
                        refs_by_part[best_part].append(child_id)

                ordered_atom_ids: list[str] = []
                for atom_id in all_atom_ids:
                    ordered_atom_ids.extend(replacements.get(atom_id, [atom_id]))
                for index, atom_id in enumerate(ordered_atom_ids):
                    connection.execute(
                        "UPDATE epub_atoms SET ordinal = ? WHERE document_id = ? AND atom_id = ?",
                        (-(index + 1), document_id, atom_id),
                    )
                for index, atom_id in enumerate(ordered_atom_ids):
                    connection.execute(
                        "UPDATE epub_atoms SET ordinal = ? WHERE document_id = ? AND atom_id = ?",
                        (index, document_id, atom_id),
                    )
                part_refs = [tuple(items) for items in refs_by_part]
            else:
                part_refs = [() for _ in parts]

            translation = self._effective_translation(row)
            has_translation = bool(translation)
            if has_translation and not (reset_translation or preserve_translation):
                raise ValueError("本段已有译文；拆分前请确认清空译文，或选择保留译文")
            preserved_part_index = (
                max(range(len(parts)), key=lambda index: len(parts[index]))
                if has_translation and preserve_translation
                else -1
            )

            existing_ids = [
                str(item["id"])
                for item in connection.execute(
                    "SELECT id FROM segments WHERE document_id = ? ORDER BY ordinal",
                    (document_id,),
                ).fetchall()
            ]
            original_index = existing_ids.index(segment_id)
            self._clear_segment_derivatives(connection, [segment_id])

            first_translation = translation if preserved_part_index == 0 else None
            connection.execute(
                """UPDATE segments SET source_text = ?, kind = ?, machine_translation = NULL,
                edited_translation = ?, reviewed_translation = NULL,
                accepted_translation = NULL, status = ?, source_refs_json = ?,
                segmentation_confidence = 1.0, segmentation_reason = 'manual_split',
                segmenter_version = 'manual-v2' WHERE id = ?""",
                (
                    parts[0],
                    SegmentKind.HEADING.value
                    if selected_as_heading and selected_part_index == 0
                    else row["kind"],
                    first_translation,
                    SegmentStatus.MACHINE_TRANSLATED.value
                    if first_translation
                    else SegmentStatus.SOURCE.value,
                    json.dumps(part_refs[0], ensure_ascii=False),
                    segment_id,
                ),
            )
            part_ids = [segment_id]
            for index, (part, source_refs) in enumerate(
                zip(parts[1:], part_refs[1:], strict=True), start=1
            ):
                new_segment_id = new_id("seg")
                part_ids.append(new_segment_id)
                part_translation = translation if preserved_part_index == index else None
                connection.execute(
                    """INSERT INTO segments
                    (id, document_id, stable_key, ordinal, kind, source_text, heading_path,
                     machine_translation, edited_translation, reviewed_translation,
                     accepted_translation, status, source_refs_json, segmentation_confidence,
                     segmentation_reason, segmenter_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?, 1.0,
                            'manual_split', 'manual-v2')""",
                    (
                        new_segment_id,
                        document_id,
                        f"{row['stable_key']}:split:{new_segment_id}",
                        -(len(existing_ids) + index + 1),
                        SegmentKind.HEADING.value
                        if selected_as_heading and selected_part_index == index
                        else row["kind"],
                        part,
                        row["heading_path"],
                        part_translation,
                        SegmentStatus.MACHINE_TRANSLATED.value
                        if part_translation
                        else SegmentStatus.SOURCE.value,
                        json.dumps(source_refs, ensure_ascii=False),
                    ),
                )
            for part_id, source_refs in zip(part_ids, part_refs, strict=True):
                if source_refs:
                    placeholders = ",".join("?" for _ in source_refs)
                    connection.execute(
                        f"""UPDATE epub_atoms SET segment_id = ?
                        WHERE document_id = ? AND atom_id IN ({placeholders})""",
                        (part_id, document_id, *source_refs),
                    )

            ordered_ids = (
                existing_ids[:original_index]
                + part_ids
                + existing_ids[original_index + 1 :]
            )
            self._renumber_segments(connection, document_id, ordered_ids)
            selected_id = part_ids[selected_part_index]
            translation_was_reset = has_translation and reset_translation
            translation_was_preserved = has_translation and preserve_translation
            self._audit(
                connection,
                "segment",
                segment_id,
                "split",
                {
                    "parts": part_ids,
                    "selected_segment_id": selected_id,
                    "translation_reset": translation_was_reset,
                    "translation_preserved": translation_was_preserved,
                    "translation_preserved_segment_id": (
                        part_ids[preserved_part_index]
                        if translation_was_preserved
                        else None
                    ),
                    "selected_as_heading": selected_as_heading,
                },
            )
        selected = self.get_segment(selected_id)
        return {
            "segment": selected,
            "segment_count": len(ordered_ids),
            "created_segment_ids": part_ids[1:],
            "translation_reset": translation_was_reset,
            "translation_preserved": translation_was_preserved,
            "translation_needs_review": translation_was_preserved,
        }

    def merge_segment(self, segment_id: str, *, direction: str) -> dict[str, Any]:
        """Merge an adjacent compatible translation unit, preserving target text for review."""
        if direction not in {"previous", "next"}:
            raise ValueError("合并方向必须是上一段或下一段")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                """SELECT s.*, d.source_format FROM segments s
                JOIN documents d ON d.id = s.document_id WHERE s.id = ?""",
                (segment_id,),
            ).fetchone()
            if current is None:
                raise NotFoundError(f"Segment not found: {segment_id}")
            self._ensure_no_active_structure_job(connection, str(current["document_id"]))
            neighbor_ordinal = int(current["ordinal"]) + (-1 if direction == "previous" else 1)
            neighbor = connection.execute(
                "SELECT * FROM segments WHERE document_id = ? AND ordinal = ?",
                (current["document_id"], neighbor_ordinal),
            ).fetchone()
            if neighbor is None:
                raise ValueError("该方向没有可合并的相邻段落")
            left, right = (neighbor, current) if direction == "previous" else (current, neighbor)
            if left["kind"] == SegmentKind.HEADING.value or right["kind"] == SegmentKind.HEADING.value:
                raise ValueError("标题不能与正文段落合并")
            if left["kind"] != right["kind"] or left["heading_path"] != right["heading_path"]:
                raise ValueError("只能合并同一章节内类型相同的相邻段落")

            left_blocks = self._source_blocks_for_row(connection, left)
            right_blocks = self._source_blocks_for_row(connection, right)
            left_refs = tuple(json.loads(left["source_refs_json"] or "[]"))
            right_refs = tuple(json.loads(right["source_refs_json"] or "[]"))
            combined_refs = left_refs + right_refs
            if current["source_format"] == "epub":
                if not left_refs or not right_refs:
                    raise ValueError("这些 EPUB 段落没有可安全合并的结构边界")
                atom_rows = connection.execute(
                    """SELECT atom_id, spine_index, ordinal FROM epub_atoms
                    WHERE document_id = ? AND atom_id IN ({}) ORDER BY ordinal""".format(
                        ",".join("?" for _ in combined_refs)
                    ),
                    (current["document_id"], *combined_refs),
                ).fetchall()
                atom_ids = [str(item["atom_id"]) for item in atom_rows]
                atom_ordinals = [int(item["ordinal"]) for item in atom_rows]
                contiguous = not atom_ordinals or atom_ordinals == list(
                    range(atom_ordinals[0], atom_ordinals[0] + len(atom_ordinals))
                )
                if (
                    atom_ids != list(combined_refs)
                    or len({item["spine_index"] for item in atom_rows}) != 1
                    or not contiguous
                ):
                    raise ValueError("EPUB 只能合并同一页面内连续的原始段落")

            merged_source = "\n\n".join(left_blocks + right_blocks)
            translations = [
                value
                for value in (
                    self._effective_translation(left),
                    self._effective_translation(right),
                )
                if value
            ]
            merged_translation = "\n\n".join(translations) or None
            document_id = str(current["document_id"])
            existing_ids = [
                str(item["id"])
                for item in connection.execute(
                    "SELECT id FROM segments WHERE document_id = ? ORDER BY ordinal",
                    (document_id,),
                ).fetchall()
            ]
            keeper_id, removed_id = str(left["id"]), str(right["id"])
            self._clear_segment_derivatives(connection, [keeper_id, removed_id])
            if combined_refs:
                placeholders = ",".join("?" for _ in combined_refs)
                connection.execute(
                    f"""UPDATE epub_atoms SET segment_id = ?
                    WHERE document_id = ? AND atom_id IN ({placeholders})""",
                    (keeper_id, document_id, *combined_refs),
                )
            connection.execute("DELETE FROM segments WHERE id = ?", (removed_id,))
            connection.execute(
                """UPDATE segments SET source_text = ?, machine_translation = NULL,
                edited_translation = ?, reviewed_translation = NULL,
                accepted_translation = NULL, status = ?, source_refs_json = ?,
                segmentation_confidence = 1.0, segmentation_reason = 'manual_merge',
                segmenter_version = 'manual-v1' WHERE id = ?""",
                (
                    merged_source,
                    merged_translation,
                    SegmentStatus.MACHINE_TRANSLATED.value
                    if merged_translation
                    else SegmentStatus.SOURCE.value,
                    json.dumps(combined_refs, ensure_ascii=False),
                    keeper_id,
                ),
            )
            ordered_ids = [item for item in existing_ids if item != removed_id]
            self._renumber_segments(connection, document_id, ordered_ids)
            self._audit(
                connection,
                "segment",
                keeper_id,
                "merged",
                {
                    "removed_segment_id": removed_id,
                    "direction": direction,
                    "translation_preserved_for_review": bool(merged_translation),
                },
            )
        merged = self.get_segment(keeper_id)
        return {
            "segment": merged,
            "segment_count": len(ordered_ids),
            "removed_segment_id": removed_id,
            "translation_needs_review": bool(merged_translation),
        }


    def get_neighbors(self, document_id: str, ordinal: int, radius: int = 1) -> list[Segment]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM segments WHERE document_id = ? AND ordinal BETWEEN ? AND ?
                ORDER BY ordinal""",
                (document_id, max(0, ordinal - radius), ordinal + radius),
            ).fetchall()
        return [self._segment(row) for row in rows]

    def add_term(self, term: TermEntry) -> TermEntry:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO terms
                (id, project_id, source, target, status, scope, domain, rationale,
                 forbidden_targets_json, aliases_json, context_keywords_json, sense,
                 disambiguation, created_at, enforcement)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    term.id,
                    term.project_id,
                    term.source,
                    term.target,
                    term.status.value,
                    term.scope,
                    term.domain,
                    term.rationale,
                    json.dumps(term.forbidden_targets, ensure_ascii=False),
                    json.dumps(term.aliases, ensure_ascii=False),
                    json.dumps(term.context_keywords, ensure_ascii=False),
                    term.sense,
                    term.disambiguation,
                    term.created_at,
                    term.enforcement,
                ),
            )
            self._audit(
                connection,
                "term",
                term.id,
                "created",
                {"source": term.source, "target": term.target, "status": term.status.value},
            )
        return term

    def update_term_enforcement(
        self, project_id: str, term_id: str, enforcement: TermEnforcement,
    ) -> TermEntry:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM terms WHERE id = ? AND project_id = ?",
                (term_id, project_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Term not found: {term_id}")
            term = self._term(row)
            if term.enforcement == enforcement:
                return term
            connection.execute(
                "UPDATE terms SET enforcement = ? WHERE id = ? AND project_id = ?",
                (enforcement, term_id, project_id),
            )
            self._audit(
                connection, "term", term_id, "enforcement_updated",
                {"before": term.enforcement, "after": enforcement},
            )
        return replace(term, enforcement=enforcement)

    def list_terms(self, project_id: str) -> list[TermEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM terms WHERE project_id = ? ORDER BY source", (project_id,)
            ).fetchall()
        return [self._term(row) for row in rows]

    def create_job(self, job: Job) -> Job:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO jobs
                (id, document_id, recipe_json, status, next_ordinal, total_cost_usd,
                 last_error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.id,
                    job.document_id,
                    json.dumps(job.recipe.to_dict(), ensure_ascii=False),
                    job.status.value,
                    job.next_ordinal,
                    job.total_cost_usd,
                    job.last_error,
                    job.created_at,
                    job.updated_at,
                ),
            )
            self._audit(connection, "job", job.id, "created", job.recipe.to_dict())
        return job

    def get_job(self, job_id: str) -> Job:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Job not found: {job_id}")
        return self._job(row)

    def save_job(self, job: Job) -> Job:
        updated = replace(job, updated_at=utc_now())
        with self._connect() as connection:
            connection.execute(
                """UPDATE jobs SET status = ?, next_ordinal = ?, total_cost_usd = ?,
                last_error = ?, updated_at = ? WHERE id = ?""",
                (
                    updated.status.value,
                    updated.next_ordinal,
                    updated.total_cost_usd,
                    updated.last_error,
                    updated.updated_at,
                    updated.id,
                ),
            )
        return updated

    def set_job_status(self, job_id: str, status: JobStatus, last_error: str | None = None) -> Job:
        job = self.get_job(job_id)
        return self.save_job(replace(job, status=status, last_error=last_error))

    def pause_interrupted_jobs(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE status = ?",
                (JobStatus.PAUSED.value, utc_now(), JobStatus.RUNNING.value),
            )

    def record_batch(
        self,
        *,
        job_id: str,
        stage: CandidateStage,
        start_ordinal: int,
        end_ordinal: int,
        segment_count: int,
        result: TranslationResult,
        elapsed_seconds: float,
    ) -> str:
        batch_id = new_id("batch")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO job_batches
                (id, job_id, stage, start_ordinal, end_ordinal, segment_count,
                 prompt_tokens, completion_tokens, reasoning_tokens,
                 prompt_cache_hit_tokens, prompt_cache_miss_tokens, cost_usd,
                 elapsed_seconds, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    batch_id,
                    job_id,
                    stage.value,
                    start_ordinal,
                    end_ordinal,
                    segment_count,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.reasoning_tokens,
                    result.prompt_cache_hit_tokens,
                    result.prompt_cache_miss_tokens,
                    result.cost_usd,
                    max(0.0, elapsed_seconds),
                    utc_now(),
                ),
            )
        return batch_id

    def job_progress(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        document_segments = self.list_segments(job.document_id)
        scoped_segments = [
            segment
            for segment in document_segments
            if not job.recipe.segment_ranges
            or any(start <= segment.ordinal <= end for start, end in job.recipe.segment_ranges)
        ]
        with self._connect() as connection:
            total_segments = len(scoped_segments)
            batch = connection.execute(
                """SELECT COUNT(*) AS batches, COALESCE(SUM(segment_count), 0) AS batch_segments,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                COALESCE(SUM(prompt_cache_hit_tokens), 0) AS cache_hit_tokens,
                COALESCE(SUM(prompt_cache_miss_tokens), 0) AS cache_miss_tokens,
                COALESCE(SUM(elapsed_seconds), 0) AS elapsed_seconds
                FROM job_batches WHERE job_id = ?""",
                (job_id,),
            ).fetchone()
            legacy = connection.execute(
                """SELECT COUNT(*) AS batches, COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                COALESCE(SUM(prompt_cache_hit_tokens), 0) AS cache_hit_tokens,
                COALESCE(SUM(prompt_cache_miss_tokens), 0) AS cache_miss_tokens
                FROM candidates WHERE job_id = ?
                AND (prompt_tokens + completion_tokens + reasoning_tokens) > 0""",
                (job_id,),
            ).fetchone()
            deferred = connection.execute(
                """SELECT COUNT(DISTINCT segment_id) AS deferred_segments
                FROM issues WHERE job_id = ? AND code = 'translation_deferred'
                AND resolved = 0""",
                (job_id,),
            ).fetchone()
            values = dict(batch)
            for key in (
                "batches",
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cache_hit_tokens",
                "cache_miss_tokens",
            ):
                values[key] = int(values[key] or 0) + int(legacy[key] or 0)
            if not values["batch_segments"]:
                values["batch_segments"] = job.next_ordinal
        processed = sum(segment.ordinal < job.next_ordinal for segment in scoped_segments)
        elapsed = float(values["elapsed_seconds"] or 0.0)
        measured = int(values["batch_segments"] or 0)
        eta_seconds = (
            max(0, total_segments - processed) * elapsed / measured
            if elapsed and measured
            else None
        )
        return asdict(job) | {
            "total_segments": total_segments,
            "processed_segments": processed,
            "batch_count": int(values["batches"] or 0),
            "prompt_tokens": int(values["prompt_tokens"] or 0),
            "completion_tokens": int(values["completion_tokens"] or 0),
            "reasoning_tokens": int(values["reasoning_tokens"] or 0),
            "cache_hit_tokens": int(values["cache_hit_tokens"] or 0),
            "cache_miss_tokens": int(values["cache_miss_tokens"] or 0),
            "total_tokens": int(values["prompt_tokens"] or 0)
            + int(values["completion_tokens"] or 0),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "deferred_segments": int(deferred["deferred_segments"] or 0),
        }

    def record_candidate(
        self,
        *,
        job_id: str,
        segment_id: str,
        stage: CandidateStage,
        provider: str,
        model: str,
        result: TranslationResult,
    ) -> str:
        candidate_id = new_id("cand")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO candidates
                (id, job_id, segment_id, stage, provider, model, text, prompt_tokens,
                 completion_tokens, reasoning_tokens, prompt_cache_hit_tokens,
                 prompt_cache_miss_tokens, cost_usd, raw_response, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) """,
                (
                    candidate_id,
                    job_id,
                    segment_id,
                    stage.value,
                    provider,
                    model,
                    result.text,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.reasoning_tokens,
                    result.prompt_cache_hit_tokens,
                    result.prompt_cache_miss_tokens,
                    result.cost_usd,
                    result.raw_response,
                    utc_now(),
                ),
            )
        return candidate_id

    def replace_issues(
        self,
        job_id: str,
        segment_id: str,
        issues: list[QualityIssue],
        *,
        target_text: str = "",
        detector_version: str = "1",
        replace_codes: set[str] | None = None,
    ) -> None:
        """Supersede prior findings and store the segment's current quality snapshot."""
        target_hash = hashlib.sha256(target_text.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            where = "segment_id = ? AND resolved = 0"
            parameters: list = [segment_id]
            if replace_codes is not None:
                codes = sorted(replace_codes)
                where += f" AND code IN ({','.join('?' for _ in codes)})"
                parameters.extend(codes)
            connection.execute(f"UPDATE issues SET resolved = 1 WHERE {where}", parameters)
            connection.executemany(
                """INSERT INTO issues
                (id, job_id, segment_id, code, message, severity, details_json,
                 detector_version, target_hash, resolved, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                [
                    (
                        new_id("issue"),
                        job_id,
                        segment_id,
                        issue.code,
                        issue.message,
                        issue.severity.value,
                        json.dumps(issue.details, ensure_ascii=False),
                        detector_version,
                        target_hash,
                        utc_now(),
                    )
                    for issue in issues
                ],
            )

    def set_machine_translation(self, segment_id: str, translation: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE segments SET machine_translation = ?, status = ? WHERE id = ?""",
                (translation, SegmentStatus.MACHINE_TRANSLATED.value, segment_id),
            )

    def set_reviewed_translation(self, segment_id: str, translation: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE segments SET reviewed_translation = ?, status = ?
                WHERE id = ?""",
                (translation, SegmentStatus.MACHINE_TRANSLATED.value, segment_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"Segment not found: {segment_id}")

    def save_segment_draft(self, segment_id: str, translation: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE segments SET edited_translation = ?, reviewed_translation = NULL WHERE id = ?",
                (translation, segment_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"Segment not found: {segment_id}")
            self._audit(
                connection,
                "segment",
                segment_id,
                "draft_saved",
                {"characters": len(translation)},
            )

    def confirm_segment(
        self, segment_id: str, translation: str, rationale: str = "", actor: str = "human"
    ) -> None:
        decision_id = new_id("decision")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source_row = connection.execute(
                """SELECT s.source_text, d.project_id FROM segments s
                JOIN documents d ON d.id = s.document_id WHERE s.id = ?""",
                (segment_id,),
            ).fetchone()
            if source_row is None:
                raise NotFoundError(f"Segment not found: {segment_id}")
            connection.execute(
                """UPDATE segments SET accepted_translation = ?, edited_translation = ?,
                status = ? WHERE id = ?""",
                (translation, translation, SegmentStatus.HUMAN_CONFIRMED.value, segment_id),
            )
            connection.execute(
                """INSERT INTO decisions
                (id, segment_id, translation, rationale, actor, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (decision_id, segment_id, translation, rationale, actor, utc_now()),
            )
            source_normalized = self._normalise_tm_source(source_row["source_text"])
            if source_normalized:
                tm_id = new_id("tm")
                updated_at = utc_now()
                connection.execute(
                    """INSERT INTO tm_entries
                    (id, project_id, source_text, source_normalized, target_text,
                     source_segment_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, source_normalized) DO UPDATE SET
                        source_text = excluded.source_text,
                        target_text = excluded.target_text,
                        source_segment_id = excluded.source_segment_id,
                        updated_at = excluded.updated_at""",
                    (
                        tm_id,
                        source_row["project_id"],
                        source_row["source_text"],
                        source_normalized,
                        translation,
                        segment_id,
                        updated_at,
                    ),
                )
            self._audit(
                connection,
                "segment",
                segment_id,
                "confirmed",
                {"decision_id": decision_id, "actor": actor, "rationale": rationale},
            )

    def search_translation_memory(
        self,
        project_id: str,
        source_text: str,
        *,
        threshold: float = 0.78,
        limit: int = 3,
    ) -> list[TranslationMemoryMatch]:
        if not source_text.strip() or limit <= 0:
            return []
        threshold = min(1.0, max(0.0, threshold))
        query = self._normalise_tm_source(source_text)
        if not query:
            return []
        with self._connect() as connection:
            exact = connection.execute(
                """SELECT * FROM tm_entries
                WHERE project_id = ? AND source_normalized = ? LIMIT 1""",
                (project_id, query),
            ).fetchone()
            rows = connection.execute(
                "SELECT * FROM tm_entries WHERE project_id = ?", (project_id,)
            ).fetchall()

        matches: list[TranslationMemoryMatch] = []
        if exact is not None:
            matches.append(self._tm_match(exact, 1.0))
        if len(matches) >= limit:
            return matches

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            candidate = row["source_normalized"]
            if candidate == query:
                continue
            max_possible = (2 * min(len(query), len(candidate))) / max(
                1, len(query) + len(candidate)
            )
            if max_possible < threshold:
                continue
            similarity = SequenceMatcher(None, query, candidate, autojunk=False).ratio()
            if similarity >= threshold:
                scored.append((similarity, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        matches.extend(self._tm_match(row, score) for score, row in scored[: limit - len(matches)])
        return matches

    def list_translation_memory(self, project_id: str) -> list[TranslationMemoryMatch]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tm_entries WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
        return [self._tm_match(row, 1.0) for row in rows]

    def get_meta(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO app_meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )

    def list_issues(self, document_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT i.*, s.ordinal FROM issues i JOIN segments s ON s.id = i.segment_id
                WHERE s.document_id = ? AND i.resolved = 0
                AND i.target_hash = sha256(COALESCE(
                    s.accepted_translation, s.reviewed_translation,
                    s.edited_translation, s.machine_translation, ''
                ))
                ORDER BY s.ordinal, i.created_at""",
                (document_id,),
            ).fetchall()
        return [dict(row) | {"details": json.loads(row["details_json"])} for row in rows]

    def list_candidates(self, segment_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM candidates WHERE segment_id = ? ORDER BY created_at", (segment_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def list_audit_events(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM audit_events WHERE entity_type = ? AND entity_id = ?
                ORDER BY created_at""",
                (entity_type, entity_id),
            ).fetchall()
        return [dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows]

    def record_provider_failure(self, segment_id: str, payload: dict[str, Any]) -> str:
        """Persist a sanitized provider-failure envelope for later diagnosis."""
        audit_id = new_id("audit")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO audit_events
                (id, entity_type, entity_id, action, payload_json, created_at)
                VALUES (?, 'segment', ?, 'provider_failure', ?, ?)""",
                (
                    audit_id,
                    segment_id,
                    json.dumps(payload, ensure_ascii=False),
                    utc_now(),
                ),
            )
        return audit_id

    def export_document(self, document_id: str, bilingual: bool = False) -> str:
        lines: list[str] = []
        for segment in self.list_segments(document_id):
            target = (
                segment.accepted_translation
                or segment.reviewed_translation
                or segment.edited_translation
                or segment.machine_translation
                or ""
            )
            if bilingual:
                lines.extend([segment.source_text, target])
            else:
                lines.append(target)
        return "\n\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        entity_type: str,
        entity_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            """INSERT INTO audit_events
            (id, entity_type, entity_id, action, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                new_id("audit"),
                entity_type,
                entity_id,
                action,
                json.dumps(payload, ensure_ascii=False),
                utc_now(),
            ),
        )

    @staticmethod
    def _normalise_tm_source(text: str) -> str:
        normalised = unicodedata.normalize("NFKC", text).casefold()
        return " ".join(part for part in re.split(r"[^\w]+", normalised) if part)

    @staticmethod
    def _tm_match(row: sqlite3.Row, similarity: float) -> TranslationMemoryMatch:
        return TranslationMemoryMatch(
            id=row["id"],
            project_id=row["project_id"],
            source_text=row["source_text"],
            target_text=row["target_text"],
            similarity=similarity,
            source_segment_id=row["source_segment_id"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _project(row: sqlite3.Row) -> Project:
        return Project(**dict(row))

    @staticmethod
    def _document(row: sqlite3.Row) -> Document:
        return Document(**dict(row))

    @staticmethod
    def _segment(row: sqlite3.Row) -> Segment:
        value = dict(row)
        value["kind"] = SegmentKind(value["kind"])
        value["status"] = SegmentStatus(value["status"])
        value["source_refs"] = tuple(json.loads(value.pop("source_refs_json", "[]")))
        return Segment(**value)

    @staticmethod
    def _term(row: sqlite3.Row) -> TermEntry:
        value = dict(row)
        value["status"] = TermStatus(value["status"])
        value["forbidden_targets"] = tuple(json.loads(value.pop("forbidden_targets_json")))
        value["aliases"] = tuple(json.loads(value.pop("aliases_json", "[]")))
        value["context_keywords"] = tuple(json.loads(value.pop("context_keywords_json", "[]")))
        return TermEntry(**value)

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        value = dict(row)
        value["recipe"] = TranslationRecipe.from_dict(json.loads(value.pop("recipe_json")))
        value["status"] = JobStatus(value["status"])
        return Job(**value)
