#!/usr/bin/env python3
"""Safely move valid, non-random legacy runs into the new hierarchy.

Use --dry_run to print planned moves. A conflict anywhere aborts the entire
operation before any move is made, so an invocation is safe to retry.
"""

import argparse
import json
import os
import os.path as osp
import shutil
import sys


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from shot_otta.artifacts import experiment_output_root  # noqa: E402


def _load_candidate(summary_path):
    run_dir = osp.dirname(summary_path)
    config_path = osp.join(run_dir, "config.yaml")
    if not osp.isfile(config_path):
        return None, "missing config.yaml"
    try:
        with open(summary_path, "r", encoding="utf-8") as file_obj:
            summary = json.load(file_obj)
        from shot_otta.config import load_yaml

        config = load_yaml(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"unreadable metadata: {error}"
    if summary.get("status") != "completed":
        return None, "summary is not completed"
    if config.get("variant") == "module_random":
        return None, "previous random baseline results are intentionally excluded"
    required = ("data", "output", "task_name", "seed", "variant")
    if any(key not in config for key in required):
        return None, "not future-compatible: incomplete effective config"
    try:
        destination_root = experiment_output_root(config)
    except (KeyError, TypeError, ValueError) as error:
        return None, f"not future-compatible: {error}"
    return (run_dir, config), None


def find_moves(source_root):
    moves, skipped = [], []
    for directory, _, filenames in os.walk(source_root):
        if "summary.json" not in filenames:
            continue
        candidate, reason = _load_candidate(osp.join(directory, "summary.json"))
        if candidate is None:
            skipped.append((directory, reason))
        else:
            moves.append(candidate)
    return moves, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="legacy runs root")
    parser.add_argument("--dst", required=True, help="new runs root")
    parser.add_argument("--dry_run", action="store_true", help="print planned moves only")
    args = parser.parse_args()
    source_root = osp.abspath(args.src)
    destination_root = osp.abspath(args.dst)
    if not osp.isdir(source_root):
        raise SystemExit(f"Source directory does not exist: {source_root}")
    moves, skipped = find_moves(source_root)
    normalized_moves = []
    for source, config in moves:
        standard_root = experiment_output_root(config)
        configured_root = osp.abspath(config["output"]["root"])
        suffix = osp.relpath(standard_root, configured_root)
        normalized_moves.append(
            (source, osp.join(destination_root, suffix, osp.basename(source)))
        )
    conflicts = [
        (source, destination)
        for source, destination in normalized_moves
        if osp.exists(destination)
    ]
    seen_destinations = set()
    for source, destination in normalized_moves:
        if destination in seen_destinations:
            conflicts.append((source, destination))
        seen_destinations.add(destination)
        parent = osp.dirname(destination)
        while parent and parent != osp.dirname(parent):
            if osp.exists(parent) and not osp.isdir(parent):
                conflicts.append((source, parent))
                break
            if parent == destination_root:
                break
            parent = osp.dirname(parent)
    for source, reason in skipped:
        print(f"SKIP {source}: {reason}")
    for source, destination in normalized_moves:
        print(f"MOVE {source} -> {destination}")
    if conflicts:
        print("Conflicts found; no files were moved:", file=sys.stderr)
        for source, destination in conflicts:
            print(f"CONFLICT {source} -> {destination}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"Dry run: {len(normalized_moves)} move(s) planned.")
        return 0
    for source, destination in normalized_moves:
        os.makedirs(osp.dirname(destination), exist_ok=True)
        shutil.move(source, destination)
    print(f"Moved {len(normalized_moves)} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
