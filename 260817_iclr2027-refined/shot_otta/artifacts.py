"""Stable run directories and lossless structured artifact writers."""

import hashlib
import json
import os
import os.path as osp
import platform
import subprocess
import sys
from datetime import datetime, timezone

from .config import dump_yaml
from protocol_constants import PROTOCOL_REVISION, SOURCE_CHECKPOINT_REVISION


SCHEMA_VERSION = 9


def json_default(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def dump_json(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(
            payload,
            file_obj,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )


def append_jsonl(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=json_default,
            )
            + "\n"
        )


def create_run_dir(output_root, task_name, run_name=None):
    os.makedirs(output_root, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    suffix = run_name.strip().replace(" ", "_") if run_name else task_name
    base_run_id = f"{timestamp}_{suffix}"
    candidate = base_run_id
    duplicate_index = 0
    while osp.exists(osp.join(output_root, candidate)):
        duplicate_index += 1
        candidate = f"{base_run_id}__dup{duplicate_index}"
    output_dir = osp.join(output_root, candidate)
    os.makedirs(output_dir)
    return candidate, output_dir


def experiment_output_root(config):
    """Return the stable hierarchy for all future experiment outputs."""
    variant = config["variant"]
    if config["method"] == "IST":
        return osp.join(
            config["output"]["root"],
            config["data"]["dataset"],
            config["task_name"],
            "IST",
            variant,
            f"seed_{int(config['seed'])}",
        )
    if variant == "source_only":
        selected_module, budget, method = "none", "none", "source_only"
    elif variant == "full_dense":
        selected_module, budget, method = "netF_netB", "full", "dense"
    elif variant == "module_dense":
        selected_module, budget, method = "netB_bottleneck", "full", "dense"
    elif variant == "conv_module_dense":
        selected_module, budget, method = "netF_layer4_conv", "full", "dense"
    elif variant.startswith("conv_"):
        selected_module = osp.join(
            "netF_layer4_conv", config["group_mode"]
        )
        budget = str(config["requested_budget"])
        method = {
            "conv_out_random": "random",
            "conv_out_magnitude": "magnitude",
            "conv_out_saliency": "saliency",
            "conv_out_lbi": "LBI",
            "conv_filter_random": "random",
            "conv_filter_magnitude": "magnitude",
            "conv_filter_saliency": "saliency",
            "conv_filter_lbi": "LBI",
        }[variant]
    else:
        selected_module = "netB_bottleneck"
        budget = str(config["requested_budget"])
        method = {
            "module_random": "random",
            "module_magnitude": "magnitude",
            "module_saliency": "saliency",
            "module_lbi": "LBI",
        }[variant]
    return osp.join(
        config["output"]["root"],
        config["data"]["dataset"],
        config["task_name"],
        selected_module,
        budget,
        method,
        f"seed_{int(config['seed'])}",
    )


def git_info(workspace_root):
    def run_git(*arguments):
        completed = subprocess.run(
            ["git", *arguments],
            cwd=workspace_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run_git("rev-parse", "HEAD")
    branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    dirty_output = run_git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if dirty_output is None else bool(dirty_output),
    }


def config_sha256(config):
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_initial_artifacts(
    output_dir,
    run_id,
    config,
    data_order,
    source_checkpoint_paths,
    selection_stats,
    workspace_root,
    summary_filename="summary.json",
):
    config_path = osp.join(output_dir, "config.yaml")
    manifest_path = osp.join(output_dir, "manifest.json")
    dump_yaml(config_path, config)

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception:
        torch_version = None
        cuda_version = None

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": config["method"],
        "variant": config["variant"],
        "task": config["task"],
        "dataset": config["data"]["dataset"],
        "source": config["data"]["source"],
        "target": config["data"]["target"],
        "source_name": config["data"]["source_name"],
        "target_name": config["data"]["target_name"],
        "seed": config["seed"],
        "implementation_revision": config["implementation_revision"],
        "protocol_revision": config.get("protocol_revision", PROTOCOL_REVISION),
        "conv_protocol_revision": config.get("conv_protocol_revision"),
        "source_checkpoint_revision": config.get(
            "source_checkpoint_revision", SOURCE_CHECKPOINT_REVISION
        ),
        "experiment_key": config["experiment_key"],
        "experiment_config_sha256": config[
            "experiment_config_sha256"
        ],
        "efficiency_protocol_revision": config.get("runtime", {}).get(
            "efficiency_protocol_revision"
        ),
        "runtime_comparable": config.get("runtime", {}).get(
            "runtime_comparable", False
        ),
        "scientific_config": config["scientific_config"],
        "command": sys.argv,
        "effective_config_sha256": config_sha256(config),
        "git": git_info(workspace_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch_version,
            "cuda": cuda_version,
        },
        "reproducibility": {
            "deterministic_cudnn": True,
            "cudnn_benchmark": False,
            "dataloader_order": data_order,
        },
        "source_checkpoints": source_checkpoint_paths,
        **selection_stats,
    }
    dump_json(manifest_path, manifest)
    return {
        "config": config_path,
        "manifest": manifest_path,
        "metrics": osp.join(output_dir, "metrics.jsonl"),
        "summary": osp.join(output_dir, summary_filename),
    }
