"""Version-pinned corpus manifests and explicit leakage checks.

Groups are supplied provenance, not inferred semantic independence. Exact
duplicate detection does not establish the absence of near-duplicate text.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .archive import ObservationArchive
from .schema import canonical_json, fields, integer, text_field

CORPUS_SCHEMA = "ai2-open-corpus-v1"
SPLITS = frozenset({"train", "validation", "test"})


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def freeze_corpus(archive: ObservationArchive, assignments: Any) -> dict[str, Any]:
    if type(assignments) is not list or not 1 <= len(assignments) <= 10_000:
        raise ValueError("corpus requires 1..10000 explicit source assignments")
    members = []
    groups: dict[str, str] = {}
    contents: dict[str, str] = {}
    seen = set()
    namespace = None
    with archive._transaction():
        for assignment in assignments:
            assignment = fields(assignment, {"source_id", "source_version", "split"})
            integer(assignment["source_version"], "source version", minimum=1)
            split = assignment["split"]
            if type(split) is not str or split not in SPLITS:
                raise ValueError("invalid corpus split")
            source = archive.verify_source(
                assignment["source_id"], assignment["source_version"]
            )
            key = (source.source_id, source.version)
            if key in seen:
                raise ValueError("duplicate source revision in corpus")
            seen.add(key)
            if namespace is not None and namespace != source.namespace:
                raise ValueError("one corpus cannot mix namespaces")
            namespace = source.namespace
            for mapping, key_value, description in (
                (groups, source.group_id, "source family"),
                (contents, source.sha256, "identical source content"),
            ):
                if key_value in mapping and mapping[key_value] != split:
                    raise ValueError(f"{description} leaks between corpus splits")
                mapping[key_value] = split
            annotations = []
            offset = 0
            while page := archive.annotations(
                source.source_id, source.version, offset=offset
            ):
                for item in page:
                    if item["binding"] is not None:
                        annotations.append(
                            {
                                "mention_id": item["mention"]["mention_id"],
                                "binding_version": item["binding"]["version"],
                            }
                        )
                offset += len(page)
            members.append(
                {
                    "source_id": source.source_id,
                    "source_version": source.version,
                    "source_sha256": source.sha256,
                    "group_id": source.group_id,
                    "metadata_sha256": _fingerprint(source.metadata),
                    "split": split,
                    "annotations": annotations,
                }
            )
    members.sort(key=lambda item: (item["source_id"], item["source_version"]))
    payload = {"namespace": namespace, "members": members}
    return {
        "schema": CORPUS_SCHEMA,
        "fingerprint": _fingerprint(payload),
        "payload": payload,
    }


def validate_corpus(archive: ObservationArchive, manifest: Any) -> dict[str, Any]:
    manifest = fields(manifest, {"schema", "fingerprint", "payload"})
    if manifest["schema"] != CORPUS_SCHEMA:
        raise ValueError("unsupported corpus schema")
    payload = fields(manifest["payload"], {"namespace", "members"})
    if _fingerprint(payload) != manifest["fingerprint"]:
        raise ValueError("corpus fingerprint mismatch")
    members = payload["members"]
    if type(members) is not list or not 1 <= len(members) <= 10_000:
        raise ValueError("invalid corpus membership")
    assignments = []
    for member in members:
        fields(
            member,
            {
                "source_id",
                "source_version",
                "source_sha256",
                "group_id",
                "metadata_sha256",
                "split",
                "annotations",
            },
        )
        assignments.append(
            {key: member[key] for key in ("source_id", "source_version", "split")}
        )
    # This also rechecks explicit source groups and identical-content leakage.
    current = freeze_corpus(archive, assignments)
    if current["payload"]["namespace"] != payload["namespace"]:
        raise ValueError("corpus namespace mismatch")
    current_sources = {
        (m["source_id"], m["source_version"]): m for m in current["payload"]["members"]
    }
    pinned_annotations = 0
    for member in members:
        current_member = current_sources[
            (member["source_id"], member["source_version"])
        ]
        for key in ("source_sha256", "group_id", "metadata_sha256"):
            if member[key] != current_member[key]:
                raise ValueError("corpus source identity mismatch")
        if type(member["annotations"]) is not list:
            raise ValueError("invalid pinned annotations")
        seen_mentions = set()
        for item in member["annotations"]:
            fields(item, {"mention_id", "binding_version"})
            text_field(item["mention_id"], "mention ID")
            integer(item["binding_version"], "binding version", minimum=1)
            if item["mention_id"] in seen_mentions:
                raise ValueError("duplicate pinned mention")
            seen_mentions.add(item["mention_id"])
            mention = archive.get_mention(item["mention_id"])
            if (mention.source_id, mention.source_version) != (
                member["source_id"],
                member["source_version"],
            ):
                raise ValueError("pinned mention belongs to another source revision")
            if archive.get_binding(mention.mention_id, item["binding_version"]) is None:
                raise ValueError("missing pinned annotation version")
            pinned_annotations += 1
    return {
        "status": "validated",
        "sources": len(members),
        "pinned_annotations": pinned_annotations,
        "fingerprint": manifest["fingerprint"],
        "near_duplicate_independence_verified": False,
    }
