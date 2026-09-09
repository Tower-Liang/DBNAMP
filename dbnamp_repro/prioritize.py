from __future__ import annotations

from typing import Any


def rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable transparent ranking used by the public wrapper.

    Similarity is primary, followed by structure confidence, evidence score,
    and identifier.  Missing numeric fields are treated as zero.
    """
    def key(row: dict[str, Any]):
        return (-float(row.get("tanimoto", 0) or 0), -float(row.get("structure_confidence", 0) or 0),
                -float(row.get("evidence_score", 0) or 0), str(row.get("id", row.get("sequence_id", ""))))
    return sorted((dict(r) for r in rows), key=key)
