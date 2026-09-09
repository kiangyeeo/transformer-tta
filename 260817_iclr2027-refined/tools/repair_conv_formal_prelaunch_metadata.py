#!/usr/bin/env python3
"""Repair only formal-plan provenance, while enforcing immutable identities."""

import copy
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_identity import build_experiment_identity
from protocol_constants import (
    CONV_IMPLEMENTATION_REVISION,
    CONV_PROTOCOL_REVISION,
    SOURCE_CHECKPOINT_REVISION,
)
from shot_otta.config import apply_overrides, load_yaml, resolve_effective_config
from train import build_parser


BASELINE_PROTOCOL = "OTTA_CONV_BASELINE_FORMAL_20260827_v1"
OFFICE_ROOT = PROJECT_ROOT / "experiment_logs/office_conv_baselines_seed2026_formal_20260827"
VISDA_ROOT = PROJECT_ROOT / "experiment_logs/visda_conv_baselines_seed2026_formal_20260827"
AUDIT_ROOT = PROJECT_ROOT / "experiment_logs/conv_baseline_formal_prelaunch_audit_20260827"
PLAN_PATHS = (
    OFFICE_ROOT / "plans/node0_ad/plan.json",
    OFFICE_ROOT / "plans/node1_aw/plan.json",
    OFFICE_ROOT / "plans/node2_da/plan.json",
    OFFICE_ROOT / "plans/node3_dw/plan.json",
    OFFICE_ROOT / "plans/node4_wa/plan.json",
    OFFICE_ROOT / "plans/node5_wd/plan.json",
    VISDA_ROOT / "plans/node6_out/plan.json",
    VISDA_ROOT / "plans/node7_filter/plan.json",
)
IMMUTABLE_FIELDS = (
    "experiment_key", "experiment_config_sha256", "command_args",
    "expected_output_root", "dataset", "source", "target", "seed",
    "variant", "requested_budget", "group_mode", "selection_seed",
    "selection_seed_mode", "num_random_masks", "child_mask_seeds",
    "random_child_mask_indices",
)


def fail(message):
    raise RuntimeError(message)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mapping_and_immutable():
    mapping, immutable = {}, {}
    for path in PLAN_PATHS:
        plan = load_json(path)
        for entry in plan.get("experiments", []):
            key, sha = entry.get("experiment_key"), entry.get("experiment_config_sha256")
            if not isinstance(key, str) or not isinstance(sha, str):
                fail(f"missing key/SHA in {path}")
            if key in mapping:
                fail(f"duplicate experiment key before repair: {key}")
            mapping[key] = sha
            immutable[key] = {field: copy.deepcopy(entry.get(field)) for field in IMMUTABLE_FIELDS}
    if len(mapping) != 133:
        fail(f"expected 133 scientific conditions before repair, found {len(mapping)}")
    return mapping, immutable


def write_mapping(path, mapping):
    ordered = {key: mapping[key] for key in sorted(mapping)}
    write_json(path, ordered)


def add_formal_field_to_json(path, remove_legacy=False):
    value = load_json(path)
    if not isinstance(value, dict):
        fail(f"planning JSON must be an object: {path}")
    if remove_legacy:
        legacy = value.pop("baseline_protocol_revision", None)
        if legacy not in (None, BASELINE_PROTOCOL):
            fail(f"unexpected legacy baseline provenance in {path}: {legacy!r}")
    value["formal_baseline_protocol_revision"] = BASELINE_PROTOCOL
    for entry in value.get("experiments", []):
        if remove_legacy:
            legacy = entry.pop("baseline_protocol_revision", None)
            if legacy not in (None, BASELINE_PROTOCOL):
                fail(f"unexpected legacy experiment provenance in {path}: {legacy!r}")
        entry["formal_baseline_protocol_revision"] = BASELINE_PROTOCOL
    write_json(path, value)


def add_formal_field_to_jsonl(path, remove_legacy=False):
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if remove_legacy:
            legacy = entry.pop("baseline_protocol_revision", None)
            if legacy not in (None, BASELINE_PROTOCOL):
                fail(f"unexpected legacy experiment provenance in {path}: {legacy!r}")
        entry["formal_baseline_protocol_revision"] = BASELINE_PROTOCOL
        entries.append(entry)
    path.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8")


def update_yaml_formal_field(path, remove_legacy=False):
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        fail(f"planning YAML must be a mapping: {path}")
    if remove_legacy:
        legacy = value.pop("baseline_protocol_revision", None)
        if legacy not in (None, BASELINE_PROTOCOL):
            fail(f"unexpected legacy matrix provenance in {path}: {legacy!r}")
    value["formal_baseline_protocol_revision"] = BASELINE_PROTOCOL
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def update_node1_generator(path):
    text = path.read_text(encoding="utf-8")
    marker = 'RATIOS = (0.0005, 0.001, 0.002)'
    if 'BASELINE_PROTOCOL = "OTTA_CONV_BASELINE_FORMAL_20260827_v1"' not in text:
        text = text.replace(marker, 'BASELINE_PROTOCOL = "OTTA_CONV_BASELINE_FORMAL_20260827_v1"\n' + marker)
    marker = 'def validate_matrix(matrix):\n'
    check = '    require(matrix["formal_baseline_protocol_revision"] == BASELINE_PROTOCOL, "formal baseline protocol revision drifted")\n'
    if check not in text:
        text = text.replace(marker, marker + check)
    entry_marker = '    return {\n        "implementation_revision": effective["implementation_revision"],'
    entry_replacement = '    return {\n        "formal_baseline_protocol_revision": BASELINE_PROTOCOL,\n        "implementation_revision": effective["implementation_revision"],'
    if entry_marker in text:
        text = text.replace(entry_marker, entry_replacement)
    plan_marker = '    plan = {"plan_schema_version": 1, "formal_evaluation": "office_aw_conv_baselines_seed2026",'
    plan_replacement = '    plan = {"plan_schema_version": 1, "formal_evaluation": "office_aw_conv_baselines_seed2026", "formal_baseline_protocol_revision": BASELINE_PROTOCOL,'
    if plan_marker in text:
        text = text.replace(plan_marker, plan_replacement)
    metadata_marker = '    metadata = {"formal_evaluation": plan["formal_evaluation"],'
    metadata_replacement = '    metadata = {"formal_evaluation": plan["formal_evaluation"], "formal_baseline_protocol_revision": BASELINE_PROTOCOL,'
    if metadata_marker in text:
        text = text.replace(metadata_marker, metadata_replacement)
    path.write_text(text, encoding="utf-8")


def update_node3_generator(path):
    text = path.read_text(encoding="utf-8")
    text = text.replace('matrix.get("baseline_protocol_revision")', 'matrix.get("formal_baseline_protocol_revision")')
    text = text.replace('"baseline_protocol_revision": BASELINE_PROTOCOL_REVISION', '"formal_baseline_protocol_revision": BASELINE_PROTOCOL_REVISION')
    path.write_text(text, encoding="utf-8")


def canonical_node6_identity(entry):
    command = entry.get("command_args")
    if not isinstance(command, list) or len(command) < 3 or command[0] != "python":
        fail(f"Node 6 command_args are invalid for {entry.get('experiment_key')}")
    args = build_parser().parse_args(command[2:])
    effective = apply_overrides(load_yaml(args.config), args)
    effective = resolve_effective_config(effective, str(WORKSPACE_ROOT))
    identity = build_experiment_identity(effective)
    if identity["experiment_key"] != entry["experiment_key"]:
        fail(f"Node 6 canonical key mismatch: {entry['experiment_key']}")
    if identity["experiment_config_sha256"] != entry["experiment_config_sha256"]:
        fail(f"Node 6 canonical SHA mismatch: {entry['experiment_key']}")
    return effective, identity


def repair_node6():
    path = VISDA_ROOT / "plans/node6_out/plan.json"
    plan = load_json(path)
    if plan.get("experiment_count") != 10 or len(plan.get("experiments", [])) != 10:
        fail("Node 6 plan must contain exactly ten experiments")
    for entry in plan["experiments"]:
        effective, identity = canonical_node6_identity(entry)
        entry.update({
            "protocol_revision": effective["protocol_revision"],
            "implementation_revision": effective["implementation_revision"],
            "protocol_track": effective["protocol_track"],
            "formal_baseline_protocol_revision": BASELINE_PROTOCOL,
            "source_checkpoint_revision": effective["source_checkpoint_revision"],
            "scientific_config": identity["scientific_config"],
        })
    plan["formal_baseline_protocol_revision"] = BASELINE_PROTOCOL
    plan["protocol_revision"] = CONV_PROTOCOL_REVISION
    plan["implementation_revision"] = CONV_IMPLEMENTATION_REVISION
    plan["source_checkpoint_revision"] = SOURCE_CHECKPOINT_REVISION
    write_json(path, plan)


def verify_after(before_mapping, before_immutable):
    after_mapping, after_immutable = mapping_and_immutable()
    if after_mapping != before_mapping:
        changed = sorted(
            key for key in set(before_mapping) | set(after_mapping)
            if before_mapping.get(key) != after_mapping.get(key)
        )
        fail(f"key->SHA mapping changed: {changed[:5]}")
    if after_immutable != before_immutable:
        changed = sorted(
            key for key in before_immutable
            if before_immutable[key] != after_immutable.get(key)
        )
        fail(f"immutable plan fields changed: {changed[:5]}")
    return after_mapping


def main():
    before_mapping, before_immutable = mapping_and_immutable()
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    before_path = AUDIT_ROOT / "before_key_to_sha256.json"
    if before_path.exists() and load_json(before_path) != {key: before_mapping[key] for key in sorted(before_mapping)}:
        fail("existing before key->SHA export does not match current plans")
    write_mapping(before_path, before_mapping)

    node1 = OFFICE_ROOT / "plans/node1_aw"
    update_yaml_formal_field(node1 / "matrix.yaml")
    for path in sorted(node1.glob("*.json")):
        add_formal_field_to_json(path)
    add_formal_field_to_jsonl(node1 / "plan.jsonl")
    update_node1_generator(node1 / "generate_plan.py")

    node3 = OFFICE_ROOT / "plans/node3_dw"
    update_yaml_formal_field(node3 / "matrix.yaml", remove_legacy=True)
    for path in sorted(node3.glob("*.json")):
        add_formal_field_to_json(path, remove_legacy=True)
    add_formal_field_to_jsonl(node3 / "plan.jsonl", remove_legacy=True)
    update_node3_generator(node3 / "generate_plan.py")

    repair_node6()
    after_mapping = verify_after(before_mapping, before_immutable)
    write_mapping(AUDIT_ROOT / "after_key_to_sha256.json", after_mapping)
    print(json.dumps({"valid": True, "condition_count": len(after_mapping), "sha_changed": 0}))


if __name__ == "__main__":
    main()
