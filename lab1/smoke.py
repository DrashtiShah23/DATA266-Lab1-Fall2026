"""Phase 1 infrastructure smoke test.

The command checks the repository foundation. Task data, models, training,
and evaluation are listed as pending until later phases implement them.
Pending task stages are not reported as passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from lab1.config import load_config
from lab1.experiment_log import append_log_record, create_raw_log, relative_log_path
from lab1.git_info import git_commit_hash
from lab1.hardware import collect_hardware_report
from lab1.layout import missing_directories
from lab1.manifest import build_manifest, write_manifest
from lab1.memory import measure_python_peak_bytes, read_peak_rss_bytes
from lab1.outputs import ensure_output_dir
from lab1.paths import repo_root, resolve_repo_path
from lab1.runtime import measure_runtime
from lab1.seed import seed_everything

PENDING_TASK_STAGES: tuple[tuple[str, str], ...] = (
    ("data_loading", "Task data loaders are not implemented."),
    ("preprocessing", "Preprocessing is not implemented."),
    ("model_construction", "Models are not implemented."),
    ("forward_pass", "No model is available for a forward pass."),
    ("loss_calculation", "No loss module is implemented."),
    ("one_training_step", "Training is not implemented."),
    ("output_creation", "Task output generation is not implemented."),
    ("evaluation", "Evaluation code is not implemented."),
)


def run_infrastructure_smoke(repo: Path, config_file: str = "configs/phase1_smoke.json") -> dict[str, Any]:
    """Run the infrastructure smoke test inside ``repo``."""
    missing = missing_directories(repo)
    if missing:
        raise FileNotFoundError("Required directories are missing: " + ", ".join(missing))
    config = load_config(config_file, repo=repo)
    for directory_field in (
        "checkpoint_directory",
        "output_directory",
        "log_directory",
        "manifest_directory",
    ):
        ensure_output_dir(config[directory_field], repo=repo)
    for relative in config["dataset_paths"].values():
        dataset_dir = resolve_repo_path(relative, repo=repo)
        if not dataset_dir.is_dir():
            raise FileNotFoundError("Configured dataset directory is missing.")
    environment = resolve_repo_path(config["environment_file"], repo=repo)
    if not environment.is_file():
        raise FileNotFoundError("Configured environment file is missing.")

    seed_report = seed_everything(config["seed"])
    hardware = collect_hardware_report()
    with measure_runtime() as timing:
        with measure_python_peak_bytes() as memory:
            memory["peak_rss_bytes"] = read_peak_rss_bytes()
    git_commit = git_commit_hash(repo)
    log_path = create_raw_log(
        config["log_directory"],
        config["experiment_name"],
        config,
        git_commit,
        hardware,
        repo=repo,
    )
    append_log_record(
        log_path,
        {
            "record_type": "infrastructure_smoke",
            "seed_report": seed_report,
            "smoke_elapsed_seconds": timing["elapsed_seconds"],
            "peak_rss_bytes": memory["peak_rss_bytes"],
            "peak_python_bytes": memory["peak_python_bytes"],
            "pending_task_stages": [
                {"stage": name, "status": "PENDING", "reason": reason}
                for name, reason in PENDING_TASK_STAGES
            ],
            "checkpoint": None,
            "metrics_file": None,
            "total_training_time": None,
            "note": "Infrastructure smoke only. No training run and no metrics were produced.",
        },
    )
    manifest = build_manifest(
        member=config["member"],
        task=config["task"],
        experiment_name=config["experiment_name"],
        git_commit=git_commit,
        config_file=config_file,
        environment_file=config["environment_file"],
        raw_log=relative_log_path(log_path, repo=repo),
        checkpoint=None,
        metrics_file=None,
        outputs_directory=config["output_directory"],
        hardware=hardware,
        start_time=timing["start_time"],
        end_time=timing["end_time"],
        total_training_time=None,
    )
    manifest_path = write_manifest(config["manifest_directory"], manifest, repo=repo)
    return {
        "status": "PASS",
        "member": config["member"],
        "log_path": relative_log_path(log_path, repo=repo),
        "manifest_path": relative_log_path(manifest_path, repo=repo),
        "git_commit": git_commit,
        "pending_task_stages": [name for name, _reason in PENDING_TASK_STAGES],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when the infrastructure checks pass."""
    parser = argparse.ArgumentParser(description="Run the Phase 1 infrastructure smoke test.")
    parser.add_argument("--config", default="configs/phase1_smoke.json")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. The documented command omits this and uses the package location.",
    )
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root()
    result = run_infrastructure_smoke(root, args.config)
    print("PHASE1_SMOKE_INFRASTRUCTURE=PASS")
    print(f"member={result['member']}")
    print(f"log={result['log_path']}")
    print(f"manifest={result['manifest_path']}")
    print(f"git_commit={result['git_commit']}")
    print("pending_task_stages=" + ",".join(result["pending_task_stages"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
