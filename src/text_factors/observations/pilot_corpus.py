"""P02 corpus receipts for a pilot with a separately held sealed evaluation.

The training archive contains only the three open splits. A custodian supplies
digest-only receipts for sealed and future sources; gold labels and source text
never enter this manifest or the pilot training view. A review attestation is
evidence of a human decision, not an automatic proof of semantic independence.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from typing import Any

from .archive import ObservationArchive
from .schema import canonical_json, fields, integer, text_field

PILOT_CORPUS_SCHEMA = "ai2-pilot-corpus-v1"
PILOT_ASSIGNMENTS_SCHEMA = "ai2-pilot-corpus-assignments-v1"
PILOT_SPLITS = (
    "train",
    "development",
    "calibration",
    "sealed_test",
    "future_stream",
)
OPEN_SPLITS = frozenset(PILOT_SPLITS[:3])
RESTRICTED_SPLITS = frozenset(PILOT_SPLITS[3:])
_METADATA = (
    "origin",
    "language",
    "rights_evidence_reference",
    "access_policy_reference",
    "annotation_policy_reference",
    "grouping_evidence_reference",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _hex_digest(value: Any, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid {name}")
    return value


def group_commitment(group_id: str) -> str:
    """Stable group comparison key for a custodian's detached receipt.

    A pilot group ID must be an independently generated opaque UUIDv4. The
    digest does not anonymize low-entropy identifiers or protect source hashes.
    """
    text_field(group_id, "source family")
    try:
        parsed = uuid.UUID(group_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("pilot family ID must be an opaque UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != group_id:
        raise ValueError("pilot family ID must be an opaque UUIDv4")
    return hashlib.sha256(("ai2-pilot-group-v1\0" + group_id).encode()).hexdigest()


def normalized_sha256(value: str) -> str:
    """NFKC, casefold, NFKC, then collapse Unicode whitespace.

    This conservative duplicate check never changes source coordinates or bytes.
    """
    if type(value) is not str:
        raise ValueError("source text must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = unicodedata.normalize("NFKC", normalized.casefold())
    return hashlib.sha256(" ".join(normalized.split()).encode("utf-8")).hexdigest()


def _source_text(archive: ObservationArchive, source_id: str, version: int) -> str:
    return "".join(item.text for item in archive.iter_observations(source_id, version))


def _check_metadata(metadata: dict[str, Any], split: str) -> None:
    for name in _METADATA:
        text_field(metadata.get(name), name)
    coverage = fields(metadata.get("annotation_coverage"), {"mentions", "identity"})
    for key in ("mentions", "identity"):
        if coverage[key] not in ("complete", "partial", "unknown"):
            raise ValueError(f"invalid {key} annotation coverage")
    if split == "train" and coverage["mentions"] != "complete":
        raise ValueError("train requires complete mention annotation coverage")


def _check_restricted(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != RESTRICTED_SPLITS:
        raise ValueError("both restricted pilot splits require detached receipts")
    result = {}
    for split in PILOT_SPLITS[3:]:
        receipt = fields(value[split], {"members"})
        members = receipt["members"]
        if (
            type(members) is not list
            or len(members) > 10_000
            or (split == "sealed_test" and not members)
        ):
            raise ValueError(f"invalid {split} receipt members")
        normalized = []
        for member in members:
            item = fields(
                member,
                {"source_sha256", "normalized_sha256", "group_commitment"},
            )
            normalized.append(
                {
                    key: _hex_digest(item[key], key)
                    for key in (
                        "source_sha256",
                        "normalized_sha256",
                        "group_commitment",
                    )
                }
            )
        for key in ("source_sha256", "normalized_sha256", "group_commitment"):
            if len({item[key] for item in normalized}) != len(normalized):
                raise ValueError(f"duplicate {key} inside {split} receipt")
        result[split] = {"members": sorted(normalized, key=canonical_json)}
    return result


def screening_sha256(groups: list[str], candidate_pairs: list[dict[str, str]]) -> str:
    """Pin the declared candidate inventory, not the truth of its screening."""
    return _digest({"group_commitments": groups, "candidate_pairs": candidate_pairs})


def _check_review(value: Any, groups: dict[str, str]) -> dict[str, Any]:
    review = fields(
        value,
        {
            "status",
            "reviewer",
            "method",
            "evidence_reference",
            "reviewed_group_commitments",
            "screening_reference",
            "screening_sha256",
            "candidate_pairs",
            "reviewed_pairs",
        },
    )
    if review["status"] != "completed":
        raise ValueError("near-duplicate independence review is not completed")
    for name in ("reviewer", "method", "evidence_reference", "screening_reference"):
        text_field(review[name], name)
    reviewed = review["reviewed_group_commitments"]
    if type(reviewed) is not list or reviewed != sorted(groups):
        raise ValueError("near-duplicate review must cover every group")
    candidates = review["candidate_pairs"]
    decisions = review["reviewed_pairs"]
    if type(candidates) is not list or type(decisions) is not list:
        raise ValueError("near-duplicate screening and decisions must be lists")
    candidate_keys = []
    for value in candidates:
        pair = fields(value, {"left", "right"})
        left = _hex_digest(pair["left"], "left group commitment")
        right = _hex_digest(pair["right"], "right group commitment")
        if left not in groups or right not in groups or left >= right:
            raise ValueError("invalid near-duplicate candidate group pair")
        if groups[left] == groups[right]:
            raise ValueError("near-duplicate candidates must span pilot splits")
        candidate_keys.append((left, right))
    if candidate_keys != sorted(set(candidate_keys)):
        raise ValueError(
            "near-duplicate screening candidates must be unique and sorted"
        )
    if _hex_digest(review["screening_sha256"], "screening fingerprint") != (
        screening_sha256(reviewed, candidates)
    ):
        raise ValueError("near-duplicate screening inventory changed")
    decision_keys = []
    for value in decisions:
        decision = fields(value, {"left", "right", "decision", "evidence_reference"})
        if decision["decision"] != "independent":
            raise ValueError("related or unresolved near-duplicate pair blocks pilot")
        text_field(decision["evidence_reference"], "pair review evidence")
        decision_keys.append((decision["left"], decision["right"]))
    if decision_keys != candidate_keys:
        raise ValueError("every near-duplicate candidate needs a matching review")
    return review


def _check_member(
    archive: ObservationArchive, member: Any, namespace: str
) -> tuple[dict[str, Any], str]:
    item = fields(
        member,
        {
            "source_id",
            "source_version",
            "source_sha256",
            "normalized_sha256",
            "group_commitment",
            "metadata_sha256",
            "split",
            "annotations",
        },
        {"identity_scopes"},
    )
    if item["split"] not in OPEN_SPLITS:
        raise ValueError("sealed and future sources cannot enter the training archive")
    integer(item["source_version"], "source version", minimum=1)
    source = archive.verify_source(item["source_id"], item["source_version"])
    if source.namespace != namespace:
        raise ValueError("pilot corpus cannot mix namespaces")
    _check_metadata(source.metadata, item["split"])
    if (
        item["source_sha256"] != source.sha256
        or item["normalized_sha256"]
        != normalized_sha256(_source_text(archive, source.source_id, source.version))
        or item["group_commitment"] != group_commitment(source.group_id)
        or item["metadata_sha256"] != _digest(source.metadata)
    ):
        raise ValueError("pinned pilot source or normalization differs from archive")
    annotations = item["annotations"]
    if type(annotations) is not list or len(annotations) > 10_000:
        raise ValueError("invalid pilot annotations")
    seen = set()
    selected_ids: dict[str, str | None] = {}
    for row in annotations:
        record = fields(row, {"mention_id", "binding_version"})
        text_field(record["mention_id"], "mention ID")
        integer(record["binding_version"], "binding version", minimum=1)
        if record["mention_id"] in seen:
            raise ValueError("duplicate pinned mention")
        seen.add(record["mention_id"])
        mention = archive.get_mention(record["mention_id"])
        binding = archive.get_binding(mention.mention_id, record["binding_version"])
        if (mention.source_id, mention.source_version) != (
            source.source_id,
            source.version,
        ) or binding is None:
            raise ValueError("pinned pilot annotation belongs to another revision")
        selected_ids[mention.mention_id] = binding.selected_id
    scopes = item.get("identity_scopes", [])
    if type(scopes) is not list or len(scopes) > 10_000:
        raise ValueError("invalid pinned identity scopes")
    scope_ids = set()
    reviewed = []
    for row in scopes:
        pinned = fields(row, {"scope_id", "version"})
        integer(pinned["version"], "identity scope version", minimum=1)
        scope = archive.get_identity_scope(pinned["scope_id"], pinned["version"])
        if scope["scope_id"] in scope_ids or (
            scope["source_id"],
            scope["source_version"],
        ) != (source.source_id, source.version):
            raise ValueError("duplicate or cross-source pinned identity scope")
        scope_ids.add(scope["scope_id"])
        if not set(scope["mention_ids"]) <= seen:
            raise ValueError("identity scope refers to unpinned mention")
        for pair in scope["pairs"]:
            left = selected_ids[pair["left_mention_id"]]
            right = selected_ids[pair["right_mention_id"]]
            if (
                left is not None
                and right is not None
                and (
                    pair["label"] == "same"
                    and left != right
                    or pair["label"] == "different"
                    and left == right
                )
            ):
                raise ValueError(
                    "explicit identity pair contradicts pinned selected instance IDs"
                )
        if scope["coverage"] == "complete":
            inside = {
                mention_id
                for mention_id in seen
                if (mention := archive.get_mention(mention_id)).char_start
                >= scope["start"]
                and mention.char_end <= scope["end"]
            }
            if set(scope["mention_ids"]) != inside:
                raise ValueError("complete identity scope omits pinned mentions")
        reviewed.append(scope)
    if (
        "identity_scopes" in item
        and (source.metadata["annotation_coverage"]["identity"] == "complete")
        and not any(
            scope["coverage"] == "complete"
            and scope["start"] == 0
            and scope["end"] == source.char_count
            for scope in reviewed
        )
    ):
        raise ValueError("complete identity metadata requires a full-source pair scope")
    return item, source.sha256


def _check_payload(archive: ObservationArchive, payload: Any) -> dict[str, Any]:
    payload = fields(
        payload,
        {
            "namespace",
            "open_members",
            "restricted_receipts",
            "near_duplicate_review",
            "split_counts",
        },
    )
    namespace = text_field(payload["namespace"], "pilot namespace")
    members = payload["open_members"]
    if type(members) is not list or not 1 <= len(members) <= 10_000:
        raise ValueError("pilot requires 1..10000 open source revisions")
    receipts = _check_restricted(payload["restricted_receipts"])
    if receipts != payload["restricted_receipts"]:
        raise ValueError("restricted receipt order must be canonical")
    split_groups: dict[str, str] = {}
    split_raw: dict[str, str] = {}
    split_normalized: dict[str, str] = {}
    source_keys = set()
    for member in members:
        item, source_sha256 = _check_member(archive, member, namespace)
        source_key = item["source_id"], item["source_version"]
        if source_key in source_keys:
            raise ValueError("duplicate source revision in pilot corpus")
        source_keys.add(source_key)
        for name, mapping, key in (
            ("related source family", split_groups, item["group_commitment"]),
            ("identical source content", split_raw, source_sha256),
            ("normalized duplicate", split_normalized, item["normalized_sha256"]),
        ):
            if key in mapping and mapping[key] != item["split"]:
                raise ValueError(f"{name} leaks between pilot splits")
            mapping[key] = item["split"]
    for split in PILOT_SPLITS[3:]:
        for member in receipts[split]["members"]:
            for name, mapping, key in (
                ("related source family", split_groups, member["group_commitment"]),
                ("identical source content", split_raw, member["source_sha256"]),
                ("normalized duplicate", split_normalized, member["normalized_sha256"]),
            ):
                if key in mapping and mapping[key] != split:
                    raise ValueError(f"{name} leaks between pilot splits")
                mapping[key] = split
    _check_review(payload["near_duplicate_review"], split_groups)
    counts = {split: 0 for split in PILOT_SPLITS}
    for member in members:
        counts[member["split"]] += 1
    for split in PILOT_SPLITS[3:]:
        counts[split] = len(receipts[split]["members"])
    if payload["split_counts"] != counts:
        raise ValueError("all five pilot split counts must match pinned sources")
    return payload


def freeze_pilot_corpus(
    archive: ObservationArchive,
    assignments: Any,
    *,
    restricted_receipts: Any,
    near_duplicate_review: Any,
) -> dict[str, Any]:
    """Pin open revisions, require detached sealed receipts and human review."""
    if type(assignments) is not list or not 1 <= len(assignments) <= 10_000:
        raise ValueError("pilot requires 1..10000 open source assignments")
    members = []
    namespace = None
    with archive._transaction():
        for assignment in assignments:
            item = fields(assignment, {"source_id", "source_version", "split"})
            if item["split"] not in OPEN_SPLITS:
                raise ValueError("sealed and future sources require detached receipts")
            integer(item["source_version"], "source version", minimum=1)
            source = archive.verify_source(item["source_id"], item["source_version"])
            if namespace is not None and namespace != source.namespace:
                raise ValueError("pilot corpus cannot mix namespaces")
            namespace = source.namespace
            annotations = []
            offset = 0
            while page := archive.annotations(
                source.source_id, source.version, offset=offset
            ):
                annotations.extend(
                    {
                        "mention_id": record["mention"]["mention_id"],
                        "binding_version": record["binding"]["version"],
                    }
                    for record in page
                    if record["binding"] is not None
                )
                offset += len(page)
            identity_scopes = []
            offset = 0
            while page := archive.identity_scopes(
                source.source_id, source.version, offset=offset
            ):
                identity_scopes.extend(
                    {"scope_id": scope["scope_id"], "version": scope["version"]}
                    for scope in page
                )
                offset += len(page)
            members.append(
                {
                    "source_id": source.source_id,
                    "source_version": source.version,
                    "source_sha256": source.sha256,
                    "normalized_sha256": normalized_sha256(
                        _source_text(archive, source.source_id, source.version)
                    ),
                    "group_commitment": group_commitment(source.group_id),
                    "metadata_sha256": _digest(source.metadata),
                    "split": item["split"],
                    "annotations": annotations,
                    "identity_scopes": identity_scopes,
                }
            )
        members.sort(key=lambda item: (item["source_id"], item["source_version"]))
        receipts = _check_restricted(restricted_receipts)
        payload = {
            "namespace": namespace,
            "open_members": members,
            "restricted_receipts": receipts,
            "near_duplicate_review": near_duplicate_review,
            "split_counts": {
                split: (
                    sum(item["split"] == split for item in members)
                    if split in OPEN_SPLITS
                    else len(receipts[split]["members"])
                )
                for split in PILOT_SPLITS
            },
        }
        _check_payload(archive, payload)
    return {
        "schema": PILOT_CORPUS_SCHEMA,
        "fingerprint": _digest(payload),
        "payload": payload,
    }


def validate_pilot_corpus(archive: ObservationArchive, manifest: Any) -> dict[str, Any]:
    manifest = fields(manifest, {"schema", "fingerprint", "payload"})
    if manifest["schema"] != PILOT_CORPUS_SCHEMA:
        raise ValueError("unsupported pilot corpus schema")
    if _digest(manifest["payload"]) != manifest["fingerprint"]:
        raise ValueError("pilot corpus fingerprint mismatch")
    with archive._transaction():
        payload = _check_payload(archive, manifest["payload"])
    return {
        "status": "validated",
        "fingerprint": manifest["fingerprint"],
        "open_source_versions": len(payload["open_members"]),
        "restricted_source_receipts": {
            split: len(payload["restricted_receipts"][split]["members"])
            for split in PILOT_SPLITS[3:]
        },
        "near_duplicate_independence": "human_review_attested",
        "sealed_labels_in_manifest": False,
    }


def pilot_training_view(archive: ObservationArchive, manifest: Any) -> dict[str, Any]:
    """Give fitting code only pinned train records, never sealed gold labels."""
    validate_pilot_corpus(archive, manifest)
    documents = []
    for item in manifest["payload"]["open_members"]:
        if item["split"] != "train":
            continue
        source = archive.get_source(item["source_id"], item["source_version"])
        annotations = []
        for record in item["annotations"]:
            mention = archive.get_mention(record["mention_id"])
            binding = archive.get_binding(mention.mention_id, record["binding_version"])
            assert binding is not None
            annotations.append(
                {
                    "start": mention.char_start,
                    "end": mention.char_end,
                    "surface": mention.surface,
                    "status": binding.status,
                    "candidate_ids": list(binding.candidate_ids),
                    "selected_id": binding.selected_id,
                    "annotator": binding.annotator,
                    "evidence": binding.evidence,
                    "binding_version": binding.version,
                }
            )
        identity_scopes = []
        identity_pairs = []
        for record in item.get("identity_scopes", []):
            scope = archive.get_identity_scope(record["scope_id"], record["version"])
            identity_scopes.append(
                {
                    key: scope[key]
                    for key in (
                        "scope_id",
                        "version",
                        "start",
                        "end",
                        "coverage",
                        "mention_ids",
                        "policy_reference",
                        "annotator",
                        "evidence",
                    )
                }
            )
            identity_pairs.extend(
                {
                    **pair,
                    "scope_id": scope["scope_id"],
                    "scope_version": scope["version"],
                    "annotator": scope["annotator"],
                    "evidence": scope["evidence"],
                    "policy_reference": scope["policy_reference"],
                }
                for pair in scope["pairs"]
            )
        documents.append(
            {
                "source_id": source.source_id,
                "source_version": source.version,
                "source_sha256": source.sha256,
                "text": _source_text(archive, source.source_id, source.version),
                # Old corpus manifests may declare full identity coverage via
                # bindings without any audited pair scope. Never pass that
                # declaration to a fitting process as pair supervision.
                "annotation_coverage": {
                    "mentions": source.metadata["annotation_coverage"]["mentions"],
                    "identity": "complete"
                    if any(
                        scope["coverage"] == "complete"
                        and scope["start"] == 0
                        and scope["end"] == source.char_count
                        for scope in identity_scopes
                    )
                    else "partial"
                    if identity_scopes
                    else "unknown",
                },
                "pair_supervision": "explicit_pairs_only",
                "annotations": annotations,
                "identity_scopes": identity_scopes,
                "identity_pairs": identity_pairs,
            }
        )
    return {
        "schema": "ai2-pilot-training-view-v1",
        "corpus_fingerprint": manifest["fingerprint"],
        "documents": documents,
    }
