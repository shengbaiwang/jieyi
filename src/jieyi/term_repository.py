from __future__ import annotations

import json
from typing import Any

from jieyi.domain.models import TermEnforcement, TermEntry, TermStatus, new_id, utc_now
from jieyi.term_discovery import model_review_counts, needs_model_review


class TermRepository:
    """SQLite adapter for evidence-bound terminology discovery records."""

    def __init__(self, store):
        self.store = store

    @staticmethod
    def _run(row) -> dict[str, Any]:
        value = dict(row)
        value["config"] = json.loads(value.pop("config_json"))
        value["coverage"] = json.loads(value.pop("coverage_json"))
        return value

    def create_run(self, run: dict[str, Any], *, reuse_scan: bool = False) -> dict[str, Any]:
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if reuse_scan:
                rows = connection.execute(
                    "SELECT * FROM term_discovery_runs WHERE document_id = ? ORDER BY created_at DESC",
                    (run["document_id"],),
                ).fetchall()
                for row in rows:
                    existing = self._run(row)
                    coverage = existing["coverage"]
                    # Document source is immutable. Legacy runs predate source_hash in coverage.
                    same_source = coverage.get("source_hash", run["coverage"]["source_hash"]) == (
                        run["coverage"]["source_hash"]
                    )
                    same_scan = all(existing["config"].get(key) == run["config"].get(key) for key in (
                        "max_candidates", "max_evidence_per_candidate", "min_score",
                    ))
                    saved_scan = coverage.get("scan_completed") or (
                        coverage.get("segments_scanned", 0) > 0
                        and coverage.get("segments_scanned") == coverage.get("segments_total")
                    )
                    if same_source and same_scan and (saved_scan or existing["status"] == "running"):
                        return existing
            connection.execute(
                """INSERT INTO term_discovery_runs
                (id, document_id, status, fingerprint, config_json, coverage_json,
                 provider, model, prompt_tokens, completion_tokens, reasoning_tokens,
                 cost_usd, error, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run["id"],
                    run["document_id"],
                    run["status"],
                    run["fingerprint"],
                    json.dumps(run["config"], ensure_ascii=False),
                    json.dumps(run["coverage"], ensure_ascii=False),
                    run["provider"],
                    run["model"],
                    run["prompt_tokens"],
                    run["completion_tokens"],
                    run["reasoning_tokens"],
                    run["cost_usd"],
                    run["error"],
                    run["created_at"],
                    run["completed_at"],
                ),
            )
            self.store._audit(
                connection,
                "term_discovery_run",
                run["id"],
                "created",
                {
                    "document_id": run["document_id"],
                    "fingerprint": run["fingerprint"],
                    "config": run["config"],
                },
            )
        return run

    def update_run(self, run_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "status",
            "coverage",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cost_usd",
            "error",
            "completed_at",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported run fields: {sorted(unknown)}")
        columns = []
        parameters = []
        for name, value in values.items():
            column = "coverage_json" if name == "coverage" else name
            columns.append(f"{column} = ?")
            parameters.append(
                json.dumps(value, ensure_ascii=False) if name == "coverage" else value
            )
        with self.store._connect() as connection:
            connection.execute(
                f"UPDATE term_discovery_runs SET {', '.join(columns)} WHERE id = ?",
                (*parameters, run_id),
            )
            self.store._audit(
                connection,
                "term_discovery_run",
                run_id,
                "updated",
                values,
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.store._connect() as connection:
            row = connection.execute(
                "SELECT * FROM term_discovery_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise LookupError(f"Term discovery run not found: {run_id}")
        return self._run(row)

    def list_runs(self, document_id: str) -> list[dict[str, Any]]:
        with self.store._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM term_discovery_runs WHERE document_id = ?
                ORDER BY created_at DESC""",
                (document_id,),
            ).fetchall()
        return [self._run(row) for row in rows]

    def find_running_run(self, document_id: str, fingerprint: str) -> dict[str, Any] | None:
        with self.store._connect() as connection:
            row = connection.execute(
                """SELECT * FROM term_discovery_runs
                WHERE document_id = ? AND fingerprint = ? AND status = 'running'
                ORDER BY created_at DESC LIMIT 1""",
                (document_id, fingerprint),
            ).fetchone()
        return self._run(row) if row is not None else None

    def fail_orphaned_runs(self) -> int:
        """Close runs that cannot survive an application process restart."""
        completed_at = utc_now()
        with self.store._connect() as connection:
            rows = connection.execute(
                "SELECT id, error, coverage_json FROM term_discovery_runs WHERE status = 'running'"
            ).fetchall()
            for row in rows:
                coverage = json.loads(row["coverage_json"])
                saved = coverage.get("scan_completed", False)
                error = row["error"] or (
                    "模型复核因应用重启中断；已保存进度，可继续复核。" if saved
                    else "Scan interrupted by application restart"
                )
                connection.execute(
                    """UPDATE term_discovery_runs
                    SET status = ?, error = ?, completed_at = ? WHERE id = ?""",
                    ("partial" if saved else "failed", error, completed_at, row["id"]),
                )
                self.store._audit(
                    connection,
                    "term_discovery_run",
                    row["id"],
                    "interrupted",
                    {"error": error, "completed_at": completed_at},
                )
        return len(rows)

    def replace_candidates(
        self, run_id: str, candidates: list[dict[str, Any]], *, coverage=None,
    ) -> None:
        now = utc_now()
        with self.store._connect() as connection:
            connection.execute("DELETE FROM term_lexeme_candidates WHERE run_id = ?", (run_id,))
            for candidate in candidates:
                connection.execute(
                    """INSERT INTO term_lexeme_candidates
                    (id, run_id, lexeme_key, canonical_form, forms_json, frequency,
                     segment_frequency, risk_score, rank, candidate_type,
                     boundary_confidence, score_components_json, extraction_methods_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        candidate["id"],
                        run_id,
                        candidate["lexeme_key"],
                        candidate["canonical_form"],
                        json.dumps(candidate["forms"], ensure_ascii=False),
                        candidate["frequency"],
                        candidate["segment_frequency"],
                        candidate["risk_score"],
                        candidate["rank"],
                        candidate.get("candidate_type", "unclassified"),
                        candidate.get("boundary_confidence", 0.0),
                        json.dumps(candidate["score_components"], ensure_ascii=False),
                        json.dumps(candidate["extraction_methods"], ensure_ascii=False),
                    ),
                )
                connection.executemany(
                    """INSERT INTO term_candidate_evidence
                    (id, lexeme_id, segment_id, ordinal, source_form, quote,
                     start_offset, end_offset, heading_path, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            evidence["id"],
                            candidate["id"],
                            evidence["segment_id"],
                            evidence["ordinal"],
                            evidence["source_form"],
                            evidence["quote"],
                            evidence["start_offset"],
                            evidence["end_offset"],
                            evidence["heading_path"],
                            evidence["reason"],
                        )
                        for evidence in candidate["evidence"]
                    ],
                )
                self._insert_senses(connection, candidate, now)
            if coverage is not None:
                connection.execute(
                    "UPDATE term_discovery_runs SET coverage_json = ? WHERE id = ?",
                    (json.dumps(coverage, ensure_ascii=False), run_id),
                )
            self.store._audit(
                connection,
                "term_discovery_run",
                run_id,
                "candidates_replaced",
                {"candidate_count": len(candidates)},
            )

    @staticmethod
    def _insert_senses(connection, candidate, now):
        connection.executemany(
            """INSERT INTO term_candidate_senses
            (id, lexeme_id, sense_key, sense, concept_definition,
             proposed_target, rationale, disambiguation, confidence,
             ai_recommended, evidence_ids_json, proposer, status,
             approved_term_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
            [
                (
                    sense["id"],
                    candidate["id"],
                    sense["sense_key"],
                    sense["sense"],
                    sense["concept_definition"],
                    sense["proposed_target"],
                    sense["rationale"],
                    sense["disambiguation"],
                    sense["confidence"],
                    None
                    if sense["ai_recommended"] is None
                    else int(sense["ai_recommended"]),
                    json.dumps(sense["evidence_ids"], ensure_ascii=False),
                    sense["proposer"],
                    sense["status"],
                    now,
                )
                for sense in candidate["senses"]
            ],
        )

    @staticmethod
    def _read_senses(connection, candidate_id):
        rows = connection.execute(
            """SELECT s.*, t.enforcement AS approved_enforcement, EXISTS (
                SELECT 1 FROM audit_events a
                WHERE a.entity_type = 'term_candidate_sense' AND a.entity_id = s.id
            ) AS human_reviewed FROM term_candidate_senses s
            LEFT JOIN terms t ON t.id = s.approved_term_id
            WHERE s.lexeme_id = ? ORDER BY s.confidence DESC, s.updated_at""",
            (candidate_id,),
        ).fetchall()
        result = []
        for row in rows:
            sense = dict(row)
            sense["evidence_ids"] = json.loads(sense.pop("evidence_ids_json"))
            sense["context_keywords"] = json.loads(sense.pop("context_keywords_json"))
            sense["human_reviewed"] = bool(sense["human_reviewed"])
            if sense["ai_recommended"] is not None:
                sense["ai_recommended"] = bool(sense["ai_recommended"])
            result.append(sense)
        return result

    def claim_review(self, run_id: str) -> bool:
        """Only one request may spend model tokens on a saved run at a time."""
        with self.store._connect() as connection:
            return connection.execute(
                """UPDATE term_discovery_runs SET status = 'running', error = '', completed_at = ''
                WHERE id = ? AND status != 'running'""", (run_id,),
            ).rowcount == 1

    def checkpoint_review(self, run_id, candidates, usage, *, baseline, coverage, limit):
        """Commit new model decisions and cumulative usage in the same transaction."""
        now = utc_now()
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for candidate in candidates:
                stored = self._read_senses(connection, candidate["id"])
                has_new_decision = any(s.get("ai_recommended") is not None for s in candidate["senses"])
                if has_new_decision and needs_model_review({"senses": stored}):
                    connection.execute(
                        "DELETE FROM term_candidate_senses WHERE lexeme_id = ?", (candidate["id"],),
                    )
                    self._insert_senses(connection, candidate, now)
                    stored = self._read_senses(connection, candidate["id"])
                # Refresh in-flight data too, so a concurrent human edit is never retried.
                candidate["senses"] = stored
            counts = model_review_counts(candidates, limit)
            coverage.update({
                "model_candidates_requested": min(len(candidates), limit),
                "model_decisions": counts["model_decisions"],
                "missing_model_decisions": counts["missing_decisions"],
                "model_kept": counts["model_kept"], "model_omitted": counts["model_omitted"],
                "model_calls": baseline["coverage"].get("model_calls", 0) + usage["model_calls"],
                "invalid_model_proposals": baseline["coverage"].get("invalid_model_proposals", 0)
                    + usage["invalid_proposals"],
                "model_error": usage.get("model_error", ""),
                "review_diagnostics": (
                    baseline["coverage"].get("review_diagnostics", []) + usage.get("diagnostics", [])
                )[-100:],
            })
            connection.execute(
                """UPDATE term_discovery_runs SET coverage_json = ?, prompt_tokens = ?,
                completion_tokens = ?, reasoning_tokens = ?, cost_usd = ? WHERE id = ?""",
                (json.dumps(coverage, ensure_ascii=False),
                 *(baseline[key] + usage[key] for key in (
                     "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd",
                 )), run_id),
            )
            self.store._audit(connection, "term_discovery_run", run_id, "review_checkpoint", {
                **counts, "model_calls": coverage["model_calls"],
            })

    def list_candidates(
        self,
        document_id: str,
        *,
        run_id: str = "",
        status: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        with self.store._connect() as connection:
            if not run_id:
                latest = connection.execute(
                    """SELECT id FROM term_discovery_runs
                    WHERE document_id = ? AND (status IN ('completed', 'partial')
                    OR json_extract(coverage_json, '$.scan_completed') = 1)
                    ORDER BY created_at DESC LIMIT 1""",
                    (document_id,),
                ).fetchone()
                if latest is None:
                    return []
                run_id = latest["id"]
            parameters: list[Any] = [run_id, document_id]
            condition = ""
            if status:
                condition = (
                    "AND EXISTS (SELECT 1 FROM term_candidate_senses s "
                    "WHERE s.lexeme_id = l.id AND s.status = ?)"
                )
                parameters.append(status)
            parameters.append(limit)
            rows = connection.execute(
                f"""SELECT l.* FROM term_lexeme_candidates l
                JOIN term_discovery_runs r ON r.id = l.run_id
                WHERE l.run_id = ? AND r.document_id = ? {condition}
                ORDER BY l.rank LIMIT ?""",
                parameters,
            ).fetchall()
            result = []
            for row in rows:
                value = dict(row)
                value["forms"] = json.loads(value.pop("forms_json"))
                value["score_components"] = json.loads(value.pop("score_components_json"))
                value["extraction_methods"] = json.loads(value.pop("extraction_methods_json"))
                evidence_rows = connection.execute(
                    """SELECT * FROM term_candidate_evidence WHERE lexeme_id = ?
                    ORDER BY ordinal, start_offset""",
                    (row["id"],),
                ).fetchall()
                value["evidence"] = [dict(item) for item in evidence_rows]
                value["senses"] = self._read_senses(connection, row["id"])
                result.append(value)
        return result

    def review_sense(
        self,
        sense_id: str,
        *,
        status: str,
        proposed_target: str,
        sense: str,
        rationale: str,
        disambiguation: str,
        actor: str,
    ) -> dict[str, Any]:
        if status not in {"pending", "rejected"}:
            raise ValueError("Review status must be pending or rejected")
        now = utc_now()
        with self.store._connect() as connection:
            row = connection.execute(
                "SELECT * FROM term_candidate_senses WHERE id = ?", (sense_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Term candidate sense not found: {sense_id}")
            if row["status"] == "approved":
                raise ValueError("Approved candidates cannot be edited as pending")
            connection.execute(
                """UPDATE term_candidate_senses SET status = ?, proposed_target = ?,
                sense = ?, rationale = ?, disambiguation = ?, updated_at = ?
                WHERE id = ?""",
                (
                    status,
                    proposed_target,
                    sense,
                    rationale,
                    disambiguation,
                    now,
                    sense_id,
                ),
            )
            self.store._audit(
                connection,
                "term_candidate_sense",
                sense_id,
                "reviewed",
                {
                    "actor": actor,
                    "status": status,
                    "proposed_target": proposed_target,
                    "sense": sense,
                    "rationale": rationale,
                    "disambiguation": disambiguation,
                },
            )
        return self.get_sense(sense_id)

    def get_sense(self, sense_id: str) -> dict[str, Any]:
        with self.store._connect() as connection:
            row = connection.execute(
                """SELECT s.*, l.canonical_form, l.forms_json, l.run_id,
                r.document_id, d.project_id, t.enforcement AS approved_enforcement
                FROM term_candidate_senses s
                JOIN term_lexeme_candidates l ON l.id = s.lexeme_id
                JOIN term_discovery_runs r ON r.id = l.run_id
                JOIN documents d ON d.id = r.document_id
                LEFT JOIN terms t ON t.id = s.approved_term_id
                WHERE s.id = ?""",
                (sense_id,),
            ).fetchone()
        if row is None:
            raise LookupError(f"Term candidate sense not found: {sense_id}")
        value = dict(row)
        value["forms"] = json.loads(value.pop("forms_json"))
        value["evidence_ids"] = json.loads(value.pop("evidence_ids_json"))
        value["context_keywords"] = json.loads(value.pop("context_keywords_json"))
        if value["ai_recommended"] is not None:
            value["ai_recommended"] = bool(value["ai_recommended"])
        return value

    def approve_sense(
        self,
        sense_id: str,
        *,
        target: str,
        sense: str,
        rationale: str,
        context_keywords: tuple[str, ...],
        disambiguation: str,
        actor: str,
        enforcement: TermEnforcement = "contextual",
    ) -> tuple[TermEntry, dict[str, Any]]:
        candidate = self.get_sense(sense_id)
        if candidate["status"] == "approved":
            raise ValueError("Candidate sense is already approved")
        source = candidate["canonical_form"].strip()
        target = target.strip()
        if not source or not target:
            raise ValueError("Source and target terms cannot be blank")
        aliases = tuple(
            form.strip()
            for form in candidate["forms"]
            if form.strip() and form.casefold() != source.casefold()
        )
        term = TermEntry(
            id=new_id("term"),
            project_id=candidate["project_id"],
            source=source,
            target=target,
            status=TermStatus.APPROVED,
            scope="project",
            enforcement=enforcement,
            rationale=rationale.strip() or candidate["rationale"],
            aliases=aliases,
            context_keywords=context_keywords,
            sense=sense.strip(),
            disambiguation=disambiguation.strip(),
        )
        now = utc_now()
        with self.store._connect() as connection:
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
                    "[]",
                    json.dumps(term.aliases, ensure_ascii=False),
                    json.dumps(term.context_keywords, ensure_ascii=False),
                    term.sense,
                    term.disambiguation,
                    term.created_at,
                    term.enforcement,
                ),
            )
            connection.execute(
                """UPDATE term_candidate_senses SET status = 'approved',
                proposed_target = ?, sense = ?, rationale = ?, disambiguation = ?,
                context_keywords_json = ?, approved_term_id = ?, updated_at = ? WHERE id = ?""",
                (
                    term.target,
                    term.sense,
                    term.rationale,
                    term.disambiguation,
                    json.dumps(term.context_keywords, ensure_ascii=False),
                    term.id,
                    now,
                    sense_id,
                ),
            )
            provenance = {
                "actor": actor,
                "candidate_sense_id": sense_id,
                "run_id": candidate["run_id"],
                "document_id": candidate["document_id"],
                "evidence_ids": candidate["evidence_ids"],
                "source": term.source,
                "context_keywords": list(term.context_keywords),
                "target": term.target,
                "enforcement": term.enforcement,
            }
            self.store._audit(connection, "term", term.id, "approved_from_candidate", provenance)
            self.store._audit(
                connection,
                "term_candidate_sense",
                sense_id,
                "approved",
                provenance | {"term_id": term.id},
            )
        return term, candidate

    def revoke_approval(self, sense_id: str, *, actor: str) -> tuple[str | None, str]:
        """Return a candidate to review and remove only the constraint it created."""
        with self.store._connect() as connection:
            # Serialize repeat requests with approval/other writers.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT s.*, d.project_id FROM term_candidate_senses s
                JOIN term_lexeme_candidates l ON l.id = s.lexeme_id
                JOIN term_discovery_runs r ON r.id = l.run_id
                JOIN documents d ON d.id = r.document_id WHERE s.id = ?""",
                (sense_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"Term candidate sense not found: {sense_id}")
            if row["status"] == "pending" and row["approved_term_id"] is None:
                return None, row["project_id"]
            if row["status"] != "approved":
                raise ValueError("Only approved candidates can have approval revoked")
            term_id = row["approved_term_id"]
            term = connection.execute("SELECT * FROM terms WHERE id = ?", (term_id,)).fetchone()
            connection.execute(
                """UPDATE term_candidate_senses SET status = 'pending',
                approved_term_id = NULL, updated_at = ? WHERE id = ?""",
                (utc_now(), sense_id),
            )
            if term_id:
                connection.execute("DELETE FROM terms WHERE id = ?", (term_id,))
                self.store._audit(
                    connection, "term", term_id, "approval_revoked",
                    {"actor": actor, "candidate_sense_id": sense_id,
                     "previous_term": dict(term) if term else None},
                )
            self.store._audit(
                connection, "term_candidate_sense", sense_id, "approval_revoked",
                {"actor": actor, "term_id": term_id, "status": "pending"},
            )
        return term_id, row["project_id"]
