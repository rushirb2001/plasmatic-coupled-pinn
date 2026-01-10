#!/usr/bin/env python3
"""
Parallel parameter sweep script for PINN benchmark evaluation.

Runs 6 concurrent training jobs on 80GB A100, with dynamic job scheduling
and rich progress tracking showing epoch progress for each running job.

Usage:
    poetry run python scripts/run_parameter_sweep.py --sweep r0
    poetry run python scripts/run_parameter_sweep.py --sweep v0
    poetry run python scripts/run_parameter_sweep.py --sweep all
    poetry run python scripts/run_parameter_sweep.py --single --r0 2.3e20 --v0 40
    poetry run python scripts/run_parameter_sweep.py --sweep all --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import yaml
from rich.console import Console
from rich.live import Live
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.physics import ParameterSpace, PlasmaParameters, DomainParameters, ScalingParameters
from src.data.fdm_solver import FDMSolver, FDMConfig, get_or_generate_fdm

# Parameter sweep configurations
# Table 1: V0=40, varying R0
R0_SWEEP_V40 = {
    "V0": 40.0,
    "R0_values": [2.3e20, 2.4e20, 2.5e20, 2.6e20, 2.7e20, 2.8e20, 2.9e20, 3.0e20,
                  3.1e20, 3.2e20, 3.3e20, 3.4e20, 3.5e20, 3.6e20, 3.7e20, 3.8e20, 3.9e20]
}

# Table 2: R0=2.3e20, varying V0
V0_SWEEP_R23 = {
    "R0": 2.3e20,
    "V0_values": [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
}

# Training settings
MAX_EPOCHS = 10000
MAX_CONCURRENT = 6  # Number of parallel jobs on 80GB A100


@dataclass
class JobConfig:
    """Configuration for a single training job."""
    R0: float
    V0: float
    job_id: Optional[str] = None  # Unique ID for this job (used for experiment dir)
    physics_config_path: Optional[Path] = None
    training_config_path: Optional[Path] = None


@dataclass
class JobResult:
    """Result from a single training run."""
    R0: float
    V0: float
    run_name: Optional[str] = None
    model_class: str = "AdaptiveSequentialPeriodicPINN"
    learning_rate: float = 1e-4
    epochs: int = MAX_EPOCHS
    batch_size: int = 20000
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256, 256])
    L2_ne: Optional[float] = None
    L2_phi: Optional[float] = None
    MSE_continuity: Optional[float] = None
    MSE_poisson: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    timestamp: Optional[str] = None
    experiment_dir: Optional[str] = None
    current_epoch: int = 0  # Track current epoch for progress


def create_parameter_space(r0: float, v0: float) -> ParameterSpace:
    """Create a ParameterSpace for given R0 and V0."""
    plasma = PlasmaParameters(
        f=13.56e6,
        V0=float(v0),
        R0=float(r0),
        T_e_eV=3.0,
        m_i_amu=40.0,
        nu_m=1.0e8,
    )
    domain = DomainParameters(L=0.025, x1=0.005, x2=0.01)
    scales = ScalingParameters(
        x_ref=0.025,
        t_ref=1.0 / 13.56e6,
        n_ref=1.0e14,
        phi_ref=float(v0),
    )
    return ParameterSpace(domain=domain, plasma=plasma, scales=scales)


def generate_fdm_for_job(r0: float, v0: float, fdm_dir: Path, config: FDMConfig) -> Path:
    """Generate FDM reference data for a job. Returns path to generated file."""
    params = create_parameter_space(r0, v0)
    fdm_dir.mkdir(parents=True, exist_ok=True)

    # This will generate if not exists, or load and validate if exists
    get_or_generate_fdm(params, config=config, fdm_dir=str(fdm_dir))

    return fdm_dir / params.get_fdm_filename(nx=config.nx, n_steps=config.n_steps_per_cycle)


def generate_physics_config(r0: float, v0: float, output_dir: Path) -> Path:
    """Generate a physics YAML config for given R0 and V0."""
    r0_str = f"{r0:.1e}".replace("+", "")
    v0_str = f"{int(v0)}"

    config_name = f"r0_{r0_str}_v0_{v0_str}.yaml"
    config_path = output_dir / config_name

    f_rf = 13.56e6
    t_ref = 1.0 / f_rf

    config = {
        "physics": {
            "domain": {
                "L": 0.025,
                "x1": 0.005,
                "x2": 0.01,
            },
            "plasma": {
                "f": 13.56e6,
                "V0": float(v0),
                "R0": float(r0),
                "T_e_eV": 3.0,
                "m_i_amu": 40.0,
                "nu_m": 1.0e8,
            },
            "scales": {
                "x_ref": 0.025,
                "t_ref": t_ref,
                "n_ref": 1.0e14,
                "phi_ref": float(v0),
            }
        }
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def generate_training_config(physics_config_path: Path, output_dir: Path, job_id: str) -> Path:
    """Generate a training config that uses the given physics config."""
    physics_name = physics_config_path.stem
    config_name = f"train_{physics_name}.yaml"
    config_path = output_dir / config_name

    physics_rel_path = f"configs/physics/sweep/{physics_config_path.name}"

    # Each job gets its own unique experiment subdirectory to avoid CSV logger conflicts
    # Set output_dir to sweep folder, experiment_name creates the subdirectory
    config = {
        "auto_name": False,  # Use fixed name to avoid conflicts
        "experiment_name": job_id,
        "output_dir": "./experiments/sweep",

        "model": {
            "class_path": "src.model.AdaptiveSequentialPeriodicPINN",
            "init_args": {
                "hidden_layers": [256, 256, 256],
                "activation": "tanh",
                "num_ffm_frequencies": 2,
                "max_t_harmonic": 4,
                "exact_bc": True,
                "use_exp_ne": True,
                "normalize_residuals": False,
                "use_ema_normalization": False,
                "ema_decay": 0.9,
                "smooth_reaction_zone": False,
                "reaction_sharpness": 100.0,
                "learning_rate": 1e-4,
                "optimizer": "adam",
                "scheduler": "constant",
                "weight_decay": 0.0,
                "loss_weights": {
                    "continuity": 1.0,
                    "poisson": 1.0,
                },
                "use_adaptive_weights": True,
                "adaptive_alpha": 0.9,
                "visualize_on_train_end": True,
                "params_path": physics_rel_path,
            }
        },

        "data": {
            "batch_size": 20000,
            "num_points": 20000,
            "sampler_type": "uniform",
            "x_range": [0.0, 1.0],
            "t_range": [0.0, 1.0],
            "clamp_x": False,
            "val_grid_size": 100,
            "num_workers": 0,
        },

        "trainer": {
            "max_epochs": MAX_EPOCHS,
            "accelerator": "auto",
            "devices": 1,
            "precision": "32-true",
            # "gradient_clip_val": 1.0,  # Disabled
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 100,
            "enable_progress_bar": False,  # Disabled for parallel runs
            "enable_model_summary": False,
        },

        "use_wandb": True,
        "wandb_project": "pinn-sweep",
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def parse_epoch_from_output(line: str) -> Optional[int]:
    """Parse current epoch from training output line (legacy, not used)."""
    match = re.search(r'EPOCH_PROGRESS:(\d+)', line)
    if match:
        return int(match.group(1))
    return None


def read_epoch_from_file(experiment_dir: Path) -> Optional[int]:
    """Read current epoch from progress file (fast, non-blocking)."""
    progress_file = experiment_dir / "progress.txt"
    try:
        if progress_file.exists():
            return int(progress_file.read_text().strip())
    except (ValueError, OSError):
        pass
    return None


def run_single_job(job: JobConfig, project_root: Path) -> JobResult:
    """Run a single training job and return results."""
    result = JobResult(R0=job.R0, V0=job.V0)
    result.timestamp = datetime.now().isoformat()
    start_time = time.time()

    # Expected experiment directory based on job_id
    experiment_dir = project_root / "experiments" / "sweep" / job.job_id

    try:
        # Use python -u for unbuffered output
        cmd = [
            "poetry", "run", "python", "-u", "-m", "src.trainer", "fit",
            "--config", str(job.training_config_path)
        ]

        # Set up environment for unbuffered output and GPU
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "0"  # All jobs on same GPU
        env["WANDB_SILENT"] = "true"

        # Use Popen for non-blocking subprocess execution
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root,
            bufsize=0,  # Unbuffered
            env=env,
        )

        # Collect output for error reporting
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)

        process.wait()
        result.duration_seconds = time.time() - start_time

        if process.returncode == 0:
            result.status = "success"
            result.experiment_dir = str(experiment_dir)
            result.run_name = job.job_id

            # Extract metrics from experiment directory
            metrics = extract_metrics(str(experiment_dir))
            result.L2_ne = metrics.get("L2_ne")
            result.L2_phi = metrics.get("L2_phi")
            result.MSE_continuity = metrics.get("MSE_continuity")
            result.MSE_poisson = metrics.get("MSE_poisson")
        else:
            result.status = "failed"
            full_output = ''.join(output_lines)
            result.error = full_output[-500:] if len(full_output) > 500 else full_output

    except Exception as e:
        result.status = "error"
        result.error = str(e)
        result.duration_seconds = time.time() - start_time

    return result


def extract_metrics(experiment_dir: str) -> Dict[str, float]:
    """Extract metrics from experiment directory."""
    metrics = {}
    exp_path = Path(experiment_dir)

    # Check for metrics.json
    metrics_file = exp_path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

    return metrics


def save_sweep_json(results: List[JobResult], sweep_json_path: Path, sweep_id: str):
    """Save all results to the sweep results JSON file."""
    data = {
        "sweep_id": sweep_id,
        "runs": [],
        "metadata": {
            "created": datetime.now().isoformat(),
            "max_epochs": MAX_EPOCHS,
            "max_concurrent": MAX_CONCURRENT,
            "total_runs": len(results),
            "successful": sum(1 for r in results if r.status == "success"),
            "failed": sum(1 for r in results if r.status in ["failed", "error", "timeout"]),
        }
    }

    for result in results:
        result_dict = asdict(result)
        result_dict.pop("experiment_dir", None)
        result_dict.pop("current_epoch", None)
        data["runs"].append(result_dict)

    sweep_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_json_path, "w") as f:
        json.dump(data, f, indent=2)


def append_result_to_sweep_json(result: JobResult, sweep_json_path: Path, sweep_id: str):
    """Append a single result to the sweep JSON immediately after run completes."""
    sweep_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data or create new
    if sweep_json_path.exists():
        with open(sweep_json_path, "r") as f:
            data = json.load(f)
    else:
        data = {
            "sweep_id": sweep_id,
            "runs": [],
            "metadata": {
                "created": datetime.now().isoformat(),
                "max_epochs": MAX_EPOCHS,
                "max_concurrent": MAX_CONCURRENT,
            }
        }

    # Convert result to dict and append
    result_dict = asdict(result)
    result_dict.pop("experiment_dir", None)
    result_dict.pop("current_epoch", None)
    data["runs"].append(result_dict)

    # Update metadata
    data["metadata"]["total_runs"] = len(data["runs"])
    data["metadata"]["successful"] = sum(1 for r in data["runs"] if r.get("status") == "success")
    data["metadata"]["failed"] = sum(1 for r in data["runs"] if r.get("status") in ["failed", "error", "timeout"])
    data["metadata"]["last_updated"] = datetime.now().isoformat()

    # Write atomically (write to temp file, then rename)
    temp_path = sweep_json_path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    temp_path.rename(sweep_json_path)


def create_status_table(jobs: List[JobConfig], results: Dict[Tuple[float, float], JobResult],
                       running: Dict[Tuple[float, float], str], project_root: Path,
                       max_epochs: int) -> Table:
    """Create a rich table showing job status with epoch progress from files."""
    table = Table(title="Parameter Sweep Progress")
    table.add_column("R0", style="cyan", width=10)
    table.add_column("V0", style="cyan", width=5)
    table.add_column("Status", style="bold", width=12)
    table.add_column("Epoch", style="magenta", width=12)
    table.add_column("L2_ne", style="green", width=8)
    table.add_column("L2_phi", style="green", width=8)
    table.add_column("MSE_cont", style="yellow", width=10)
    table.add_column("MSE_pois", style="yellow", width=10)
    table.add_column("Time", width=8)

    for job in jobs:
        key = (job.R0, job.V0)
        if key in results:
            r = results[key]
            status_style = {
                "success": "[green]✓ done[/green]",
                "failed": "[red]✗ failed[/red]",
                "error": "[red]✗ error[/red]",
                "timeout": "[yellow]⏱ timeout[/yellow]",
                "pending": "[dim]pending[/dim]",
            }.get(r.status, r.status)

            l2_ne = f"{r.L2_ne*100:.2f}%" if r.L2_ne else "-"
            l2_phi = f"{r.L2_phi*100:.2f}%" if r.L2_phi else "-"
            mse_cont = f"{r.MSE_continuity:.2e}" if r.MSE_continuity else "-"
            mse_pois = f"{r.MSE_poisson:.2e}" if r.MSE_poisson else "-"
            duration = f"{r.duration_seconds:.0f}s" if r.duration_seconds else "-"
            epoch_str = f"{max_epochs}/{max_epochs}"

            table.add_row(f"{job.R0:.1e}", str(int(job.V0)), status_style, epoch_str,
                         l2_ne, l2_phi, mse_cont, mse_pois, duration)
        elif key in running:
            # Read epoch from progress file (fast, non-blocking)
            job_id = running[key]
            exp_dir = project_root / "experiments" / "sweep" / job_id
            current_epoch = read_epoch_from_file(exp_dir) or 0
            progress_pct = (current_epoch / max_epochs) * 100 if max_epochs > 0 else 0
            epoch_str = f"{current_epoch}/{max_epochs}"
            status_str = f"[yellow]⟳ {progress_pct:.0f}%[/yellow]"
            table.add_row(f"{job.R0:.1e}", str(int(job.V0)), status_str, epoch_str,
                         "-", "-", "-", "-", "-")
        else:
            table.add_row(f"{job.R0:.1e}", str(int(job.V0)), "[dim]queued[/dim]", "-",
                         "-", "-", "-", "-", "-")

    return table


def run_parallel_sweep(jobs: List[JobConfig], project_root: Path,
                       sweep_json_path: Path, sweep_id: str,
                       max_concurrent: int = MAX_CONCURRENT, dry_run: bool = False) -> List[JobResult]:
    """Run jobs in parallel with dynamic scheduling. Saves results immediately after each run."""
    console = Console()
    results: Dict[Tuple[float, float], JobResult] = {}
    running: Dict[Tuple[float, float], str] = {}  # Maps (R0, V0) -> job_id for progress file lookup

    if dry_run:
        console.print("[yellow]DRY RUN - Generating FDM reference datasets[/yellow]")
        console.print(f"Total configurations: {len(jobs)}")

        # Use benchmark FDM config for reference data generation
        fdm_config = FDMConfig.benchmark()
        fdm_dir = project_root / "data" / "fdm"

        for i, job in enumerate(jobs, 1):
            console.print(f"\n[bold][{i}/{len(jobs)}] Generating FDM for R0={job.R0:.2e}, V0={job.V0}[/bold]")
            try:
                fdm_path = generate_fdm_for_job(job.R0, job.V0, fdm_dir, fdm_config)
                console.print(f"  [green]Saved to: {fdm_path}[/green]")
                result = JobResult(R0=job.R0, V0=job.V0, status="fdm_generated")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                result = JobResult(R0=job.R0, V0=job.V0, status="fdm_error", error=str(e))
            results[(job.R0, job.V0)] = result

        console.print(f"\n[bold green]FDM generation complete![/bold green]")
        return list(results.values())

    console.print(f"\n[bold]Starting parallel sweep with {max_concurrent} concurrent jobs[/bold]")
    console.print(f"Total jobs: {len(jobs)}")
    console.print(f"Max epochs: {MAX_EPOCHS}")
    console.print()

    # Use ThreadPoolExecutor for parallel job execution
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        job_iter = iter(jobs)

        # Fill up to max_concurrent
        for _ in range(min(max_concurrent, len(jobs))):
            try:
                job = next(job_iter)
                future = executor.submit(run_single_job, job, project_root)
                futures[future] = job
                running[(job.R0, job.V0)] = job.job_id
            except StopIteration:
                break

        # Process completions and submit new jobs
        with Live(create_status_table(jobs, results, running, project_root, MAX_EPOCHS),
                  console=console, refresh_per_second=2) as live:
            while futures:
                # Wait for any future to complete
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    time.sleep(0.25)
                    live.update(create_status_table(jobs, results, running, project_root, MAX_EPOCHS))
                    continue

                for future in done_futures:
                    job = futures.pop(future)
                    running.pop((job.R0, job.V0), None)

                    try:
                        result = future.result()
                    except Exception as e:
                        result = JobResult(R0=job.R0, V0=job.V0, status="error", error=str(e))

                    results[(job.R0, job.V0)] = result

                    # Save result to sweep JSON immediately (crash-safe)
                    append_result_to_sweep_json(result, sweep_json_path, sweep_id)

                    # Submit next job if available
                    try:
                        next_job = next(job_iter)
                        next_future = executor.submit(run_single_job, next_job, project_root)
                        futures[next_future] = next_job
                        running[(next_job.R0, next_job.V0)] = next_job.job_id
                    except StopIteration:
                        pass

                live.update(create_status_table(jobs, results, running, project_root, MAX_EPOCHS))

    return list(results.values())


def prepare_jobs(sweep_type: str, config_dir: Path, sweep_id: str) -> List[JobConfig]:
    """Prepare job configurations for the sweep."""
    config_dir.mkdir(parents=True, exist_ok=True)

    if sweep_type == "r0":
        combinations = [(r0, R0_SWEEP_V40["V0"]) for r0 in R0_SWEEP_V40["R0_values"]]
    elif sweep_type == "v0":
        combinations = [(V0_SWEEP_R23["R0"], v0) for v0 in V0_SWEEP_R23["V0_values"]]
    elif sweep_type == "all":
        combinations = [(r0, R0_SWEEP_V40["V0"]) for r0 in R0_SWEEP_V40["R0_values"]]
        combinations += [(V0_SWEEP_R23["R0"], v0) for v0 in V0_SWEEP_R23["V0_values"] if v0 != 40]
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")

    jobs = []
    for r0, v0 in combinations:
        # Create unique job ID for this run
        r0_str = f"{r0:.1e}".replace("+", "").replace(".", "p")
        v0_str = f"{int(v0)}"
        job_id = f"r0_{r0_str}_v0_{v0_str}_{sweep_id}"

        physics_config = generate_physics_config(r0, v0, config_dir)
        training_config = generate_training_config(physics_config, config_dir, job_id)
        jobs.append(JobConfig(
            R0=r0,
            V0=v0,
            job_id=job_id,
            physics_config_path=physics_config,
            training_config_path=training_config,
        ))

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run parallel PINN parameter sweep")
    parser.add_argument("--sweep", choices=["r0", "v0", "all"], help="Type of sweep to run")
    parser.add_argument("--single", action="store_true", help="Run single configuration")
    parser.add_argument("--r0", type=float, default=2.3e20, help="R0 value for single run")
    parser.add_argument("--v0", type=float, default=40.0, help="V0 value for single run")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs without training")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"Number of parallel jobs (default: {MAX_CONCURRENT})")

    args = parser.parse_args()

    max_concurrent = args.max_concurrent

    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs" / "physics" / "sweep"

    # Generate unique sweep ID for this run
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_json_path = project_root / "experiments" / "sweep" / f"sweep_results_{sweep_id}.json"

    console = Console()

    if args.single:
        console.print(f"\n[bold]Single Run: R0={args.r0:.2e}, V0={args.v0}[/bold]")

        r0_str = f"{args.r0:.1e}".replace("+", "").replace(".", "p")
        v0_str = f"{int(args.v0)}"
        job_id = f"r0_{r0_str}_v0_{v0_str}_{sweep_id}"

        physics_config = generate_physics_config(args.r0, args.v0, config_dir)
        training_config = generate_training_config(physics_config, config_dir, job_id)

        console.print(f"Physics config: {physics_config}")
        console.print(f"Training config: {training_config}")

        if args.dry_run:
            console.print("[yellow]DRY RUN - Skipping training[/yellow]")
            return

        job = JobConfig(R0=args.r0, V0=args.v0,
                       physics_config_path=physics_config,
                       training_config_path=training_config)
        result = run_single_job(job, project_root)

        console.print(f"\nStatus: {result.status}")
        if result.status == "success":
            console.print(f"L2_ne: {result.L2_ne*100:.4f}%" if result.L2_ne else "L2_ne: N/A")
            console.print(f"L2_phi: {result.L2_phi*100:.4f}%" if result.L2_phi else "L2_phi: N/A")
            console.print(f"MSE_cont: {result.MSE_continuity:.4e}" if result.MSE_continuity else "MSE_cont: N/A")
            console.print(f"MSE_pois: {result.MSE_poisson:.4e}" if result.MSE_poisson else "MSE_pois: N/A")
            console.print(f"Duration: {result.duration_seconds:.0f}s" if result.duration_seconds else "")
        else:
            console.print(f"Error: {result.error}")

        save_sweep_json([result], sweep_json_path, sweep_id)
        console.print(f"Results saved to: {sweep_json_path}")

    elif args.sweep:
        console.print(f"\n[bold]Parameter Sweep: {args.sweep.upper()}[/bold]")
        console.print(f"Sweep ID: {sweep_id}")
        console.print(f"Results will be saved to: {sweep_json_path}")

        jobs = prepare_jobs(args.sweep, config_dir, sweep_id)
        console.print(f"Prepared {len(jobs)} job configurations")

        # Results are saved immediately after each run completes (crash-safe)
        results = run_parallel_sweep(
            jobs, project_root,
            sweep_json_path=sweep_json_path,
            sweep_id=sweep_id,
            max_concurrent=max_concurrent,
            dry_run=args.dry_run
        )

        # Print summary
        success = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status in ["failed", "error", "timeout"])

        console.print(f"\n[bold]Sweep Complete[/bold]")
        console.print(f"Success: {success}/{len(results)}")
        console.print(f"Failed: {failed}/{len(results)}")
        console.print(f"Results: {sweep_json_path}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
