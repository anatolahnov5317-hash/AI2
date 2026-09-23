"""Read-only discovery smoke on three pinned, public AI2 documents.

This is an import/coordinate check, not training data or a pilot evaluation.
All documents are conservatively grouped into one related project family.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text_factors.observations import ArchiveLimits, ObservationArchive  # noqa: E402

COMMIT = "d3fba8ed33a38673c6a942f1ae682c373da50d03"
DOCUMENTS = {
    "docs/ALGORITHM.md": (
        "934a9c00751d4f573c5dcab10b151a3943058ba1ffed5a231c3de386a8dab373"
    ),
    "docs/OPEN_OBSERVATIONS.md": (
        "28515dca25c6fe7a5f0f599e4152d053d7f39003d451122a145c21b00c4b45f6"
    ),
    "docs/real_data/P02_CORPUS_INTEGRITY.md": (
        "1f820d138d020f13e83af2f4cec9783320d4d8d0f82f83125815aa3c09d8e4ce"
    ),
}


def pinned_bytes(path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{COMMIT}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def main() -> None:
    license_text = pinned_bytes("LICENSE")
    if not license_text.startswith(b"MIT License\n"):
        raise ValueError("the pinned repository license changed")

    # There is only ONE family: documentation about the same AI2 project is
    # related. A fresh UUID is scoped to this ephemeral archive, not a split.
    family_id = str(uuid.uuid4())
    results: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="ai2-public-discovery-") as temp:  # noqa: SIM117
        with ObservationArchive(
            Path(temp) / "discovery.sqlite",
            create=True,
            limits=ArchiveLimits(chunk_chars=73),
        ) as archive:
            for path, expected_digest in DOCUMENTS.items():
                raw = pinned_bytes(path)
                if hashlib.sha256(raw).hexdigest() != expected_digest:
                    raise ValueError(f"pinned content differs: {path}")
                decoded = raw.decode("utf-8", errors="strict")
                if not any("А" <= char <= "я" for char in decoded):
                    raise ValueError(f"expected Russian text: {path}")
                metadata = {
                    "origin": f"https://github.com/anatolahnov5317-hash/AI2/blob/{COMMIT}/{path}",
                    "language": "ru",
                    "rights_evidence_reference": (
                        f"https://github.com/anatolahnov5317-hash/AI2/blob/{COMMIT}/LICENSE"
                    ),
                    "access_policy_reference": (
                        "public repository, discovery smoke only"
                    ),
                    "annotation_policy_reference": "none; no gold labels",
                    "grouping_evidence_reference": (
                        "same AI2 project; conservative single family"
                    ),
                    "annotation_coverage": {
                        "mentions": "unknown",
                        "identity": "unknown",
                    },
                }
                arguments = {
                    "namespace": "public-discovery-only",
                    "external_key": f"{COMMIT}:{path}",
                    "group_id": family_id,
                    "metadata": metadata,
                }
                # Small blocks exercise a UTF-8 sequence split across input
                # boundaries without changing original code-point coordinates.
                source = archive.import_blocks(
                    (raw[start : start + 19] for start in range(0, len(raw), 19)),
                    **arguments,
                )
                repeat = archive.import_blocks([raw], **arguments)
                if (source.source_id, source.version) != (
                    repeat.source_id,
                    repeat.version,
                ) or source.version != 1:
                    raise AssertionError(f"repeat import created a revision: {path}")
                if source.sha256 != expected_digest or source.group_id != family_id:
                    raise AssertionError(f"provenance changed: {path}")
                archive.verify_source(source.source_id, source.version)
                char_cursor = byte_cursor = 0
                chunks = list(
                    archive.iter_observations(source.source_id, source.version)
                )
                for chunk in chunks:
                    if (chunk.char_start, chunk.byte_start) != (
                        char_cursor,
                        byte_cursor,
                    ):
                        raise AssertionError(f"non-contiguous coordinates: {path}")
                    if chunk.text != decoded[chunk.char_start : chunk.char_end]:
                        raise AssertionError(f"character coordinates changed: {path}")
                    if chunk.raw != raw[chunk.byte_start : chunk.byte_end]:
                        raise AssertionError(f"byte coordinates changed: {path}")
                    char_cursor, byte_cursor = chunk.char_end, chunk.byte_end
                if b"".join(chunk.raw for chunk in chunks) != raw:
                    raise AssertionError(f"original bytes were not restored: {path}")
                if (char_cursor, byte_cursor) != (len(decoded), len(raw)):
                    raise AssertionError(f"wrong final coordinates: {path}")
                results.append(
                    {
                        "path": path,
                        "sha256": source.sha256,
                        "bytes": source.byte_count,
                        "codepoints": source.char_count,
                        "chunks": len(chunks),
                        "reimport_version": repeat.version,
                    }
                )
            if len(archive.sources("public-discovery-only")) != len(DOCUMENTS):
                raise AssertionError("unexpected number of sources")

    print(
        json.dumps(
            {
                "status": "passed",
                "git_commit": COMMIT,
                "rights_evidence": (
                    "AI2 repository MIT LICENSE; third-party excerpts "
                    "require separate review"
                ),
                "independent_groups_claimed": 0,
                "related_groups_in_smoke": 1,
                "annotation_coverage": "unknown",
                "documents": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
