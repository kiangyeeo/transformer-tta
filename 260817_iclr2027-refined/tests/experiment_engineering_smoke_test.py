#!/usr/bin/env python3
"""Lightweight experiment planning/status/aggregation checks."""

import copy
import csv
import json
import math
import os
import os.path as osp
import shlex
import sys
import tempfile


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
WORKSPACE_ROOT = osp.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from experiment_identity import build_experiment_identity  # noqa: E402
from shot_otta.config import (  # noqa: E402
    load_yaml,
    resolve_effective_config,
)
from tools.check_experiment_status import (  # noqa: E402
    check_status,
    scan_run_summaries,
    write_status_outputs,
)
from tools.plan_experiments import (  # noqa: E402
    _expand_transfers,
    _validate_unique_identities,
    build_plan,
    validate_matrix,
    write_plan_outputs,
)
from tools.summarize_runs import (  # noqa: E402
    aggregate_across_seeds,
    aggregate_dataset_macro,
    build_summary_outputs,
    write_csv,
    write_markdown,
    write_summary_outputs,
)


BASE_CONFIG_PATH = osp.join(
    PROJECT_DIR,
    "configs",
    "otta_fc_lbi_protocol_20260817_v1.yaml",
)
LBI_CONFIG_FIELDS = (
    "alpha",
    "kappa",
    "nu",
    "omega",
    "stage1_max_steps",
    "budget_tolerance",
    "stage2_lr",
    "stage2_steps",
    "delta_nonzero_tolerance",
    "support_threshold",
)


def _lbi_tuple(alpha=0.1, kappa=1.0, nu=1.0):
    return {
        "alpha": alpha,
        "kappa": kappa,
        "nu": nu,
        "omega": 0.1,
        "stage1_max_steps": 3000,
        "budget_tolerance": 0.0001,
        "stage2_lr": 0.01,
        "stage2_steps": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "support_threshold": 1.0e-4,
    }


def _pilot_matrix():
    return {
        "method": "shot",
        "task": "otta",
        "datasets": {"office": {"transfers": [[0, 1]]}},
        "seeds": [2026],
        "variants": {
            "source_only": {},
            "full_dense": {},
            "module_dense": {},
            "module_random": {
                "budgets": [0.001],
                "selection_seed_mode": "same_as_run_seed",
            },
            "module_magnitude": {"budgets": [0.001]},
            "module_saliency": {"budgets": [0.001]},
            "module_lbi": {"budgets": [0.001], "lbi": _lbi_tuple()},
        },
        "common_training": {"batch_size": 64, "workers": 4, "save_model": False},
        "output_root": "260817_iclr2027-refined/runs",
    }


def _lbi_fixture(round1=False):
    matrix = {
        "method": "shot",
        "task": "otta",
        "datasets": {"office": {"transfers": [[0, 1]]}},
        "seeds": [2026],
        "variants": {"module_lbi": {"budgets": [0.001], "lbi": _lbi_tuple()}},
        "common_training": {"batch_size": 64, "workers": 4, "save_model": False},
        "output_root": "260817_iclr2027-refined/runs",
    }
    if round1:
        matrix["variants"]["module_lbi"] = {
            "budgets": [0.002],
            "lbi_trials": [
                _lbi_tuple(0.15), _lbi_tuple(0.2), _lbi_tuple(0.1, 1.5),
                _lbi_tuple(0.1, 2.0), _lbi_tuple(0.1, 1.0, 0.5),
            ],
        }
    return matrix


def _effective(
    variant="module_random",
    seed=2026,
    budget=0.001,
):
    config = load_yaml(BASE_CONFIG_PATH)
    config["variant"] = variant
    config["seed"] = seed
    config["requested_budget"] = budget
    config["selection_seed"] = seed
    return resolve_effective_config(config, WORKSPACE_ROOT)


def _check_identity():
    effective = _effective()
    baseline = build_experiment_identity(effective)
    non_scientific = copy.deepcopy(effective)
    non_scientific["output"]["root"] = "/tmp/different output"
    non_scientific["output"]["run_name"] = "renamed"
    non_scientific["output"]["save_model"] = (
        not effective["output"]["save_model"]
    )
    non_scientific["device"]["gpu_id"] = "7"
    unchanged = build_experiment_identity(non_scientific)
    assert unchanged == baseline

    same_again = build_experiment_identity(copy.deepcopy(effective))
    assert same_again["experiment_key"] == baseline["experiment_key"]
    assert (
        same_again["experiment_config_sha256"]
        == baseline["experiment_config_sha256"]
    )

    changed_configs = []
    optimizer_changed = copy.deepcopy(effective)
    optimizer_changed["optimization"]["momentum"] = 0.8
    changed_configs.append(optimizer_changed)
    loss_changed = copy.deepcopy(effective)
    loss_changed["loss"]["ent_par"] = 0.75
    changed_configs.append(loss_changed)
    seed_changed = copy.deepcopy(effective)
    seed_changed["seed"] = 2025
    seed_changed["selection_seed"] = 2025
    changed_configs.append(seed_changed)
    budget_changed = copy.deepcopy(effective)
    budget_changed["requested_budget"] = 0.01
    changed_configs.append(budget_changed)
    for changed in changed_configs:
        identity = build_experiment_identity(changed)
        assert (
            identity["experiment_config_sha256"]
            != baseline["experiment_config_sha256"]
        )
        assert identity["experiment_key"] != baseline["experiment_key"]


def _check_planner():
    matrix = _pilot_matrix()
    base = load_yaml(BASE_CONFIG_PATH)
    plan = build_plan(matrix, base, BASE_CONFIG_PATH)
    assert plan["experiment_count"] == 7
    by_variant = {}
    for experiment in plan["experiments"]:
        by_variant.setdefault(experiment["variant"], []).append(
            experiment
        )
    assert all(
        len(by_variant[variant]) == 1
        for variant in (
            "source_only",
            "full_dense",
            "module_dense",
        )
    )
    assert by_variant["source_only"][0]["requested_budget"] is None
    assert by_variant["full_dense"][0]["requested_budget"] == 1.0
    assert by_variant["module_dense"][0]["requested_budget"] == 1.0
    random_entry = by_variant["module_random"][0]
    assert random_entry["selection_seed"] == random_entry["seed"]
    assert random_entry["selection_seed_mode"] == "same_as_run_seed"
    lbi_entry = by_variant["module_lbi"][0]
    assert lbi_entry["requested_budget"] == 0.001
    assert "--lbi-alpha" in lbi_entry["command_args"]
    assert "--lbi-stage2-steps" in lbi_entry["command_args"]
    assert "--lbi-support-threshold" in lbi_entry["command_args"]

    two_budget_matrix = _pilot_matrix()
    two_budget_matrix["variants"]["module_random"]["budgets"] = [
        0.001,
        0.01,
    ]
    two_budget_plan = build_plan(
        two_budget_matrix,
        load_yaml(BASE_CONFIG_PATH),
        BASE_CONFIG_PATH,
    )
    random_entries = [
        experiment
        for experiment in two_budget_plan["experiments"]
        if experiment["variant"] == "module_random"
    ]
    assert len(random_entries) == 2
    assert {
        experiment["selection_seed"]
        for experiment in random_entries
    } == {2026}

    ordered = _expand_transfers("office", "all_ordered_pairs")
    assert len(ordered) == 6
    assert all(source != target for source, target in ordered)

    duplicate_transfer = _pilot_matrix()
    duplicate_transfer["datasets"]["office"]["transfers"] = [
        [0, 1],
        [0, 1],
    ]
    try:
        validate_matrix(duplicate_transfer)
    except ValueError as error:
        assert "Duplicate transfer" in str(error)
    else:
        raise AssertionError("Duplicate transfer was not rejected")

    duplicate_seed = _pilot_matrix()
    duplicate_seed["seeds"] = [2026, 2026]
    try:
        validate_matrix(duplicate_seed)
    except ValueError as error:
        assert "Duplicate seed" in str(error)
    else:
        raise AssertionError("Duplicate seed was not rejected")

    self_transfer = _pilot_matrix()
    self_transfer["datasets"]["office"]["transfers"] = [[1, 1]]
    try:
        validate_matrix(self_transfer)
    except ValueError as error:
        assert "source == target" in str(error)
    else:
        raise AssertionError("source == target was not rejected")

    duplicate_identity_entries = [
        {
            "experiment_key": "duplicate",
            "experiment_config_sha256": "a" * 64,
        },
        {
            "experiment_key": "duplicate",
            "experiment_config_sha256": "b" * 64,
        },
    ]
    try:
        _validate_unique_identities(duplicate_identity_entries)
    except ValueError as error:
        assert "experiment_key" in str(error)
    else:
        raise AssertionError("Duplicate experiment key was not rejected")
    duplicate_hash_entries = [
        {
            "experiment_key": "first",
            "experiment_config_sha256": "c" * 64,
        },
        {
            "experiment_key": "second",
            "experiment_config_sha256": "c" * 64,
        },
    ]
    try:
        _validate_unique_identities(duplicate_hash_entries)
    except ValueError as error:
        assert "experiment_config_sha256" in str(error)
    else:
        raise AssertionError("Duplicate full hash was not rejected")

    with tempfile.TemporaryDirectory(
        prefix="iclr2027_plan_smoke_"
    ) as temp_dir:
        write_plan_outputs(plan, temp_dir)
        for filename in ("plan.json", "plan.jsonl", "commands.sh"):
            assert osp.isfile(osp.join(temp_dir, filename))
        with open(
            osp.join(temp_dir, "commands.sh"),
            "r",
            encoding="utf-8",
        ) as file_obj:
            command_lines = [
                line.strip()
                for line in file_obj
                if line.strip() and not line.startswith("#")
                and not line.startswith("set ")
            ]
        assert len(command_lines) == 7
        assert shlex.split(command_lines[0]) == (
            plan["experiments"][0]["command_args"]
        )


def _command_value(command_args, option):
    index = command_args.index(option)
    return command_args[index + 1]


def _expect_matrix_error(matrix, expected_text):
    try:
        validate_matrix(matrix)
    except ValueError as error:
        assert expected_text in str(error), str(error)
    else:
        raise AssertionError(
            f"Invalid matrix was accepted; expected {expected_text!r}"
        )


def _check_lbi_trials_planner():
    base = load_yaml(BASE_CONFIG_PATH)
    old_matrix = _lbi_fixture()
    old_plan = build_plan(
        old_matrix,
        copy.deepcopy(base),
        BASE_CONFIG_PATH,
    )
    assert old_plan["experiment_count"] == 1
    old_entry = old_plan["experiments"][0]
    assert old_entry["lbi_trial_index"] == 0

    new_matrix = _lbi_fixture()
    single_trial = new_matrix["variants"]["module_lbi"].pop("lbi")
    new_matrix["variants"]["module_lbi"]["lbi_trials"] = [
        single_trial
    ]
    new_plan = build_plan(
        new_matrix,
        copy.deepcopy(base),
        BASE_CONFIG_PATH,
    )
    new_entry = new_plan["experiments"][0]
    assert new_entry["lbi_trial_index"] == 0
    for field in (
        "experiment_key",
        "experiment_config_sha256",
        "scientific_config",
        "effective_overrides",
        "command_args",
    ):
        assert new_entry[field] == old_entry[field]
    assert "lbi_trial_index" not in new_entry["scientific_config"]
    assert "lbi_trial_index" not in new_entry["effective_overrides"]
    assert "--lbi-trial-index" not in new_entry["command_args"]

    round1_matrix = _lbi_fixture(round1=True)
    round1_plan = build_plan(
        round1_matrix,
        copy.deepcopy(base),
        BASE_CONFIG_PATH,
    )
    assert round1_plan["experiment_count"] == 5
    expected_trials = [
        (0.15, 1.0, 1.0),
        (0.2, 1.0, 1.0),
        (0.1, 1.5, 1.0),
        (0.1, 2.0, 1.0),
        (0.1, 1.0, 0.5),
    ]
    for index, (entry, expected) in enumerate(
        zip(round1_plan["experiments"], expected_trials)
    ):
        alpha, kappa, nu = expected
        assert entry["lbi_trial_index"] == index
        assert (
            entry["alpha"],
            entry["kappa"],
            entry["nu"],
        ) == expected
        override_lbi = entry["effective_overrides"]["lbi"]
        scientific_lbi = entry["scientific_config"]["lbi"]
        expected_lbi = round1_matrix["variants"]["module_lbi"][
            "lbi_trials"
        ][index]
        for field in LBI_CONFIG_FIELDS:
            assert override_lbi[field] == expected_lbi[field]
            assert scientific_lbi[field] == expected_lbi[field]
        command = entry["command_args"]
        assert float(_command_value(command, "--lbi-alpha")) == alpha
        assert float(_command_value(command, "--lbi-kappa")) == kappa
        assert float(_command_value(command, "--lbi-nu")) == nu
        assert (
            float(_command_value(command, "--lbi-omega"))
            == override_lbi["omega"]
        )
        assert (
            int(
                _command_value(
                    command,
                    "--lbi-stage1-max-steps",
                )
            )
            == override_lbi["stage1_max_steps"]
        )
        assert (
            float(
                _command_value(
                    command,
                    "--lbi-budget-tolerance",
                )
            )
            == override_lbi["budget_tolerance"]
        )
        assert (
            float(_command_value(command, "--lbi-stage2-lr"))
            == override_lbi["stage2_lr"]
        )
        assert (
            int(_command_value(command, "--lbi-stage2-steps"))
            == override_lbi["stage2_steps"]
        )
        assert (
            float(
                _command_value(
                    command,
                    "--lbi-delta-nonzero-tolerance",
                )
            )
            == override_lbi["delta_nonzero_tolerance"]
        )
        assert (
            float(_command_value(command, "--lbi-support-threshold"))
            == override_lbi["support_threshold"]
        )
    assert len(
        {
            entry["experiment_key"]
            for entry in round1_plan["experiments"]
        }
    ) == 5
    assert len(
        {
            entry["experiment_config_sha256"]
            for entry in round1_plan["experiments"]
        }
    ) == 5

    six_matrix = _lbi_fixture(round1=True)
    six_spec = six_matrix["variants"]["module_lbi"]
    six_spec["budgets"] = [0.001, 0.002]
    six_spec["lbi_trials"] = six_spec["lbi_trials"][:3]
    six_plan = build_plan(
        six_matrix,
        copy.deepcopy(base),
        BASE_CONFIG_PATH,
    )
    assert six_plan["experiment_count"] == 6
    assert [
        (
            entry["requested_budget"],
            entry["lbi_trial_index"],
        )
        for entry in six_plan["experiments"]
    ] == [
        (0.001, 0),
        (0.001, 1),
        (0.001, 2),
        (0.002, 0),
        (0.002, 1),
        (0.002, 2),
    ]

    both = _lbi_fixture()
    both["variants"]["module_lbi"]["lbi_trials"] = [
        copy.deepcopy(both["variants"]["module_lbi"]["lbi"])
    ]
    _expect_matrix_error(both, "both were provided")

    neither = _lbi_fixture()
    del neither["variants"]["module_lbi"]["lbi"]
    _expect_matrix_error(neither, "exactly one of lbi or lbi_trials")

    empty = _lbi_fixture()
    del empty["variants"]["module_lbi"]["lbi"]
    empty["variants"]["module_lbi"]["lbi_trials"] = []
    _expect_matrix_error(empty, "must be a non-empty list")

    wrong_container = _lbi_fixture()
    wrong_trial = wrong_container["variants"]["module_lbi"].pop("lbi")
    wrong_container["variants"]["module_lbi"]["lbi_trials"] = {
        "trial": wrong_trial
    }
    _expect_matrix_error(
        wrong_container,
        "must be a non-empty list",
    )

    non_mapping = _lbi_fixture()
    del non_mapping["variants"]["module_lbi"]["lbi"]
    non_mapping["variants"]["module_lbi"]["lbi_trials"] = ["invalid"]
    _expect_matrix_error(non_mapping, "must be a mapping")

    missing = _lbi_fixture()
    missing_trial = missing["variants"]["module_lbi"].pop("lbi")
    del missing_trial["alpha"]
    missing["variants"]["module_lbi"]["lbi_trials"] = [missing_trial]
    _expect_matrix_error(missing, "Missing LBI config fields")

    illegal = _lbi_fixture()
    illegal_trial = illegal["variants"]["module_lbi"].pop("lbi")
    illegal_trial["alpha"] = 0
    illegal["variants"]["module_lbi"]["lbi_trials"] = [illegal_trial]
    _expect_matrix_error(illegal, "lbi.alpha must be > 0")

    implicit_grid = _lbi_fixture()
    implicit_trial = implicit_grid["variants"]["module_lbi"].pop("lbi")
    implicit_trial["alpha"] = [0.1, 0.2]
    implicit_grid["variants"]["module_lbi"]["lbi_trials"] = [
        implicit_trial
    ]
    _expect_matrix_error(implicit_grid, "Invalid LBI trial")

    duplicate = _lbi_fixture()
    duplicate_trial = duplicate["variants"]["module_lbi"].pop("lbi")
    duplicate["variants"]["module_lbi"]["lbi_trials"] = [
        copy.deepcopy(duplicate_trial),
        copy.deepcopy(duplicate_trial),
    ]
    _expect_matrix_error(
        duplicate,
        "Duplicate LBI trial configurations at indices 0 and 1",
    )

    wrong_variant = _pilot_matrix()
    wrong_variant["variants"]["module_random"]["lbi_trials"] = [
        copy.deepcopy(single_trial)
    ]
    _expect_matrix_error(
        wrong_variant,
        "module_random must not define lbi_trials",
    )

    wrong_legacy_variant = _pilot_matrix()
    wrong_legacy_variant["variants"]["module_random"]["lbi"] = (
        copy.deepcopy(single_trial)
    )
    _expect_matrix_error(
        wrong_legacy_variant,
        "module_random must not define lbi",
    )


def _write_summary(root, dirname, payload):
    run_dir = osp.join(root, dirname)
    os.makedirs(run_dir)
    path = osp.join(run_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj)
    return path


def _planned(key, sha):
    return {
        "experiment_key": key,
        "experiment_config_sha256": sha,
        "method": "shot",
        "task": "otta",
        "dataset": "office",
        "source": 0,
        "target": 1,
        "seed": 2020,
        "variant": "source_only",
        "requested_budget": None,
        "selection_seed": None,
        "selection_seed_mode": None,
    }


def _check_status():
    planned = [
        _planned("completed", "a" * 64),
        _planned("missing", "b" * 64),
        _planned("duplicate", "c" * 64),
        _planned("mismatch", "d" * 64),
        _planned("invalid", "e" * 64),
    ]
    plan = {"experiments": planned}
    with tempfile.TemporaryDirectory(
        prefix="iclr2027_status_smoke_"
    ) as temp_dir:
        runs_root = osp.join(temp_dir, "runs")
        os.makedirs(runs_root)
        _write_summary(
            runs_root,
            "completed",
            {
                "experiment_key": "completed",
                "experiment_config_sha256": "a" * 64,
                "status": "completed",
            },
        )
        for index in range(2):
            _write_summary(
                runs_root,
                f"duplicate-{index}",
                {
                    "experiment_key": "duplicate",
                    "experiment_config_sha256": "c" * 64,
                    "status": "completed",
                },
            )
        _write_summary(
            runs_root,
            "mismatch",
            {
                "experiment_key": "mismatch",
                "experiment_config_sha256": "f" * 64,
                "status": "completed",
            },
        )
        _write_summary(
            runs_root,
            "invalid",
            {
                "experiment_key": "invalid",
                "experiment_config_sha256": "e" * 64,
            },
        )
        records, invalid = scan_run_summaries(runs_root)
        result = check_status(plan, runs_root)
        statuses = {
            row["experiment_key"]: row["plan_status"]
            for row in result["experiments"]
        }
        assert statuses == {
            "completed": "completed",
            "missing": "missing",
            "duplicate": "duplicate_completed",
            "mismatch": "hash_mismatch",
            "invalid": "invalid_summary",
        }
        assert len(records) == 4
        assert len(invalid) == 1
        output_dir = osp.join(temp_dir, "status")
        write_status_outputs(result, output_dir)
        for filename in (
            "experiment_status.json",
            "experiment_status.csv",
            "missing_runs.csv",
            "duplicate_runs.csv",
        ):
            assert osp.isfile(osp.join(output_dir, filename))


def _run_row(source, target, seed, pu, fo, runtime):
    return {
        "experiment_key": f"k-{source}-{target}-{seed}",
        "experiment_config_sha256": (
            f"{source}{target}{seed}".encode("utf-8").hex().ljust(64, "0")
        )[:64],
        "method": "shot",
        "task": "otta",
        "dataset": "office",
        "source": source,
        "target": target,
        "variant": "module_magnitude",
        "requested_budget": 0.001,
        "selection_seed": None,
        "seed": seed,
        "PU-Acc": pu,
        "FO-Acc": fo,
        "runtime": runtime,
    }


def _check_aggregation():
    rows = [
        _run_row(0, 1, 1, 0.0, 10.0, 2.0),
        _run_row(0, 1, 2, 100.0, 30.0, 4.0),
        _run_row(1, 0, 1, 0.0, 50.0, 8.0),
    ]
    transfer = aggregate_across_seeds(rows)
    first = next(
        row
        for row in transfer
        if row["source"] == 0 and row["target"] == 1
    )
    second = next(
        row
        for row in transfer
        if row["source"] == 1 and row["target"] == 0
    )
    assert first["PU-Acc_mean"] == 50.0
    assert math.isclose(
        first["PU-Acc_sample_std"],
        math.sqrt(5000.0),
    )
    assert second["PU-Acc_sample_std"] is None

    lbi_rows = []
    for alpha in (0.1, 0.2):
        row = _run_row(0, 1, 1, 12.0, 34.0, 5.0)
        row.update(
            {
                "variant": "module_lbi",
                "alpha": alpha,
                "kappa": 1.0,
                "nu": 1.0,
                "omega": 0.1,
                "stage1_max_steps": 3000,
                "budget_tolerance": 0.0001,
                "stage2_lr": 0.01,
                "stage2_steps_requested": 1,
                "delta_nonzero_tolerance": 1.0e-12,
            }
        )
        lbi_rows.append(row)
    lbi_groups = aggregate_across_seeds(lbi_rows)
    assert len(lbi_groups) == 2
    assert {row["alpha"] for row in lbi_groups} == {0.1, 0.2}

    expected_plan = {
        "experiments": [
            {
                **_run_row(0, 1, 1, None, None, None),
                "selection_seed_mode": None,
            },
            {
                **_run_row(0, 1, 2, None, None, None),
                "selection_seed_mode": None,
            },
            {
                **_run_row(1, 0, 1, None, None, None),
                "selection_seed_mode": None,
            },
            {
                **_run_row(2, 0, 1, None, None, None),
                "selection_seed_mode": None,
            },
        ]
    }
    macro = aggregate_dataset_macro(
        transfer,
        plan=expected_plan,
        completed_rows=rows,
    )
    assert len(macro) == 1
    macro_row = macro[0]
    assert macro_row["PU-Acc_macro_mean"] == 25.0
    assert macro_row["completed_transfer_count"] == 2
    assert macro_row["expected_transfer_count"] == 3
    assert macro_row["coverage_ratio"] == 2 / 3
    assert macro_row["is_complete"] is False

    precise = 0.12345678901234568
    precision_rows = [{"value": precise}]
    with tempfile.TemporaryDirectory(
        prefix="iclr2027_precision_smoke_"
    ) as temp_dir:
        csv_path = osp.join(temp_dir, "values.csv")
        markdown_path = osp.join(temp_dir, "values.md")
        json_path = osp.join(temp_dir, "values.json")
        write_csv(csv_path, precision_rows)
        write_markdown(markdown_path, precision_rows)
        with open(json_path, "w", encoding="utf-8") as file_obj:
            json.dump(precision_rows, file_obj)
        with open(csv_path, "r", encoding="utf-8") as file_obj:
            csv_value = next(csv.DictReader(file_obj))["value"]
        with open(markdown_path, "r", encoding="utf-8") as file_obj:
            markdown = file_obj.read()
        with open(json_path, "r", encoding="utf-8") as file_obj:
            json_text = file_obj.read()
        assert csv_value == str(precise)
        assert str(precise) in markdown
        assert str(precise) in json_text

    with tempfile.TemporaryDirectory(
        prefix="iclr2027_summary_smoke_"
    ) as temp_dir:
        runs_root = osp.join(temp_dir, "runs")
        os.makedirs(runs_root)
        completed = _run_row(0, 1, 1, precise, 77.0, 3.0)
        completed.update(
            {
                "status": "completed",
                "source-target": "amazon-dslr",
                "run_id": "fake-run",
            }
        )
        _write_summary(runs_root, "fake-run", completed)
        summary_plan = {
            "experiments": [
                {
                    **_run_row(0, 1, 1, None, None, None),
                    "selection_seed_mode": None,
                },
                {
                    **_run_row(1, 0, 1, None, None, None),
                    "selection_seed_mode": None,
                },
            ]
        }
        outputs = build_summary_outputs(
            runs_root,
            plan=summary_plan,
        )
        assert len(outputs["all_runs"]) == 1
        assert len(outputs["missing_runs"]) == 1
        assert (
            outputs["dataset_macro_aggregated"][0]["is_complete"]
            is False
        )
        output_dir = osp.join(temp_dir, "summary")
        write_summary_outputs(outputs, output_dir)
        expected_files = (
            "all_runs.json",
            "all_runs.csv",
            "all_runs.md",
            "per_transfer_seed_aggregated.json",
            "per_transfer_seed_aggregated.csv",
            "per_transfer_seed_aggregated.md",
            "dataset_macro_aggregated.json",
            "dataset_macro_aggregated.csv",
            "dataset_macro_aggregated.md",
            "missing_runs.csv",
            "duplicate_runs.csv",
            "invalid_runs.csv",
        )
        assert all(
            osp.isfile(osp.join(output_dir, filename))
            for filename in expected_files
        )
        with open(
            osp.join(output_dir, "all_runs.csv"),
            "r",
            encoding="utf-8",
        ) as file_obj:
            assert str(precise) in file_obj.read()


def main():
    _check_identity()
    _check_planner()
    _check_lbi_trials_planner()
    _check_status()
    _check_aggregation()
    print("iclr2027 experiment engineering smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
