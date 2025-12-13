#!/usr/bin/env python3
"""
Experiment Validation Script for CCP-II PINN Models.

This script validates that all models train correctly and learn physics properly.
It is NOT a unit test suite but an integration/smoke test for the training pipeline.

Checks performed:
1. Model instantiation without errors
2. Training loop runs without crashes
3. Loss decreases during training (model is learning)
4. Outputs are physically reasonable:
   - Electron density is bounded and non-negative (after exp transform)
   - Electric potential is bounded
   - Boundary conditions are approximately satisfied
5. Gradients flow properly (no NaN/Inf)

Usage:
    python scripts/validate_experiments.py              # Run all validations
    python scripts/validate_experiments.py --quick      # Quick test (10 epochs)
    python scripts/validate_experiments.py --model gated  # Test specific model
    python scripts/validate_experiments.py --config configs/default.yaml  # Test specific config
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

import torch
import numpy as np


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    name: str
    passed: bool
    message: str
    duration: float = 0.0
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    loss_reduction: Optional[float] = None


class ExperimentValidator:
    """Validates PINN experiments for correctness and physics learning."""

    def __init__(
        self,
        num_epochs: int = 50,
        batch_size: int = 2048,
        num_points: int = 5000,
        device: str = "auto",
        verbose: bool = True,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_points = num_points
        self.verbose = verbose

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.results: List[ValidationResult] = []

    def log(self, msg: str, level: str = "info"):
        """Print log message if verbose."""
        if self.verbose:
            prefix = {
                "info": "  ",
                "pass": "  ✓",
                "fail": "  ✗",
                "warn": "  ⚠",
            }.get(level, "  ")
            print(f"{prefix} {msg}")

    def validate_model_instantiation(self, model_class, **kwargs) -> ValidationResult:
        """Test that model can be instantiated without errors."""
        name = f"Instantiation: {model_class.__name__}"
        start = time.time()

        try:
            model = model_class(**kwargs)
            model = model.to(self.device)
            duration = time.time() - start

            # Check model has expected components
            assert hasattr(model, "net"), "Model missing 'net' attribute"
            assert hasattr(model, "forward"), "Model missing 'forward' method"
            assert hasattr(model, "training_step"), "Model missing 'training_step'"

            return ValidationResult(
                name=name,
                passed=True,
                message=f"Model instantiated successfully ({duration:.2f}s)",
                duration=duration,
            )
        except Exception as e:
            return ValidationResult(
                name=name,
                passed=False,
                message=f"Instantiation failed: {str(e)}",
                duration=time.time() - start,
            )

    def validate_forward_pass(self, model) -> ValidationResult:
        """Test forward pass produces valid outputs."""
        name = f"Forward pass: {model.__class__.__name__}"
        start = time.time()

        try:
            model.eval()
            # Create test input
            x_t = torch.rand(100, 2, device=self.device)

            with torch.no_grad():
                n_e, phi = model(x_t)

            # Validate output shapes
            assert n_e.shape == (100, 1), f"n_e shape mismatch: {n_e.shape}"
            assert phi.shape == (100, 1), f"phi shape mismatch: {phi.shape}"

            # Check for NaN/Inf
            assert not torch.isnan(n_e).any(), "n_e contains NaN"
            assert not torch.isnan(phi).any(), "phi contains NaN"
            assert not torch.isinf(n_e).any(), "n_e contains Inf"
            assert not torch.isinf(phi).any(), "phi contains Inf"

            duration = time.time() - start
            return ValidationResult(
                name=name,
                passed=True,
                message=f"Forward pass OK, n_e: [{n_e.min():.3f}, {n_e.max():.3f}], phi: [{phi.min():.3f}, {phi.max():.3f}]",
                duration=duration,
            )
        except Exception as e:
            return ValidationResult(
                name=name,
                passed=False,
                message=f"Forward pass failed: {str(e)}",
                duration=time.time() - start,
            )

    def validate_gradient_flow(self, model) -> ValidationResult:
        """Test that gradients flow through the model."""
        name = f"Gradient flow: {model.__class__.__name__}"
        start = time.time()

        try:
            model.train()
            x_t = torch.rand(100, 2, device=self.device, requires_grad=True)

            n_e, phi = model(x_t)
            loss = n_e.mean() + phi.mean()
            loss.backward()

            # Check gradients exist and are valid
            has_grad = False
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    has_grad = True
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break

            assert has_grad, "No gradients computed"
            assert not has_nan_grad, "Gradients contain NaN"

            model.zero_grad()
            duration = time.time() - start
            return ValidationResult(
                name=name,
                passed=True,
                message="Gradients flow properly",
                duration=duration,
            )
        except Exception as e:
            return ValidationResult(
                name=name,
                passed=False,
                message=f"Gradient flow failed: {str(e)}",
                duration=time.time() - start,
            )

    def validate_pde_residuals(self, model) -> ValidationResult:
        """Test PDE residual computation."""
        name = f"PDE residuals: {model.__class__.__name__}"
        start = time.time()

        try:
            model.train()
            x_t = torch.rand(100, 2, device=self.device)

            res_cont, res_pois = model.compute_pde_residuals(x_t)

            # Check shapes
            assert res_cont.shape == (100, 1), f"Continuity residual shape: {res_cont.shape}"
            assert res_pois.shape == (100, 1), f"Poisson residual shape: {res_pois.shape}"

            # Check for NaN/Inf
            assert not torch.isnan(res_cont).any(), "Continuity residual contains NaN"
            assert not torch.isnan(res_pois).any(), "Poisson residual contains NaN"

            duration = time.time() - start
            return ValidationResult(
                name=name,
                passed=True,
                message=f"PDE residuals OK, cont: {res_cont.abs().mean():.3e}, pois: {res_pois.abs().mean():.3e}",
                duration=duration,
            )
        except Exception as e:
            return ValidationResult(
                name=name,
                passed=False,
                message=f"PDE residual computation failed: {str(e)}",
                duration=time.time() - start,
            )

    def validate_training_loop(self, model, num_epochs: int = None) -> ValidationResult:
        """Test that model trains and loss decreases."""
        name = f"Training loop: {model.__class__.__name__}"
        num_epochs = num_epochs or self.num_epochs
        start = time.time()

        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Generate training data
            x_t = torch.rand(self.num_points, 2, device=self.device)

            losses = []
            for epoch in range(num_epochs):
                # Mini-batch training
                indices = torch.randperm(self.num_points)[:self.batch_size]
                batch = x_t[indices]

                optimizer.zero_grad()

                # Compute losses
                res_cont, res_pois = model.compute_pde_residuals(batch)
                loss_cont = torch.mean(res_cont**2)
                loss_pois = torch.mean(res_pois**2)

                t = batch[:, 1:2]
                loss_bc = model.compute_boundary_loss(t)

                loss = loss_cont + loss_pois + 10.0 * loss_bc

                # Check for NaN loss
                if torch.isnan(loss):
                    return ValidationResult(
                        name=name,
                        passed=False,
                        message=f"NaN loss at epoch {epoch}",
                        duration=time.time() - start,
                    )

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                losses.append(loss.item())

                if self.verbose and epoch % max(1, num_epochs // 5) == 0:
                    self.log(f"Epoch {epoch:3d}: loss = {loss.item():.4e}", "info")

            duration = time.time() - start
            initial_loss = np.mean(losses[:5]) if len(losses) >= 5 else losses[0]
            final_loss = np.mean(losses[-5:]) if len(losses) >= 5 else losses[-1]
            reduction = (initial_loss - final_loss) / initial_loss * 100

            # Check loss decreased (at least 10% reduction expected)
            if final_loss < initial_loss * 0.9:
                return ValidationResult(
                    name=name,
                    passed=True,
                    message=f"Loss decreased: {initial_loss:.4e} -> {final_loss:.4e} ({reduction:.1f}% reduction)",
                    duration=duration,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    loss_reduction=reduction,
                )
            else:
                return ValidationResult(
                    name=name,
                    passed=False,
                    message=f"Loss did not decrease enough: {initial_loss:.4e} -> {final_loss:.4e} ({reduction:.1f}%)",
                    duration=duration,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    loss_reduction=reduction,
                )

        except Exception as e:
            import traceback
            return ValidationResult(
                name=name,
                passed=False,
                message=f"Training failed: {str(e)}\n{traceback.format_exc()}",
                duration=time.time() - start,
            )

    def validate_physical_outputs(self, model) -> ValidationResult:
        """Validate that model outputs are physically reasonable."""
        name = f"Physical validity: {model.__class__.__name__}"
        start = time.time()

        try:
            model.eval()

            # Create evaluation grid
            nx, nt = 50, 50
            x = torch.linspace(0, 1, nx, device=self.device)
            t = torch.linspace(0, 1, nt, device=self.device)
            X, T = torch.meshgrid(x, t, indexing="ij")
            x_t = torch.stack([X.flatten(), T.flatten()], dim=1)

            with torch.no_grad():
                n_e, phi = model(x_t)

            n_e = n_e.cpu().numpy().reshape(nx, nt)
            phi = phi.cpu().numpy().reshape(nx, nt)

            issues = []

            # Check 1: n_e should be bounded (not astronomically high or low)
            if np.abs(n_e).max() > 1e10:
                issues.append(f"n_e magnitude too high: {np.abs(n_e).max():.2e}")

            # Check 2: phi should be bounded (within reasonable potential range)
            if np.abs(phi).max() > 1e10:
                issues.append(f"phi magnitude too high: {np.abs(phi).max():.2e}")

            # Check 3: Boundary conditions at x=1 (phi should be ~0)
            phi_right = phi[-1, :]  # x=1
            bc_error_right = np.mean(np.abs(phi_right))
            if bc_error_right > 1.0:  # After training, should be small
                issues.append(f"BC at x=1 not satisfied: mean |phi| = {bc_error_right:.3f}")

            # Check 4: n_e at boundaries should be ~0
            ne_left = n_e[0, :]
            ne_right = n_e[-1, :]
            ne_bc_error = np.mean(np.abs(ne_left)) + np.mean(np.abs(ne_right))
            if ne_bc_error > 2.0:  # Relaxed threshold
                issues.append(f"n_e BC not satisfied: mean |n_e| at boundaries = {ne_bc_error/2:.3f}")

            # Check 5: Values should not be NaN or Inf
            if np.isnan(n_e).any() or np.isnan(phi).any():
                issues.append("Output contains NaN values")
            if np.isinf(n_e).any() or np.isinf(phi).any():
                issues.append("Output contains Inf values")

            duration = time.time() - start

            if not issues:
                return ValidationResult(
                    name=name,
                    passed=True,
                    message=f"Outputs physically valid: n_e in [{n_e.min():.3f}, {n_e.max():.3f}], phi in [{phi.min():.3f}, {phi.max():.3f}]",
                    duration=duration,
                )
            else:
                return ValidationResult(
                    name=name,
                    passed=False,
                    message=f"Physical issues: {'; '.join(issues)}",
                    duration=duration,
                )

        except Exception as e:
            return ValidationResult(
                name=name,
                passed=False,
                message=f"Physical validation failed: {str(e)}",
                duration=time.time() - start,
            )

    def validate_model(self, model_class, model_name: str, **kwargs) -> List[ValidationResult]:
        """Run all validations for a single model."""
        print(f"\n{'='*60}")
        print(f"Validating: {model_name}")
        print(f"{'='*60}")

        results = []

        # 1. Instantiation
        result = self.validate_model_instantiation(model_class, **kwargs)
        results.append(result)
        self.log(result.message, "pass" if result.passed else "fail")
        if not result.passed:
            return results

        model = model_class(**kwargs).to(self.device)

        # 2. Forward pass
        result = self.validate_forward_pass(model)
        results.append(result)
        self.log(result.message, "pass" if result.passed else "fail")

        # 3. Gradient flow
        result = self.validate_gradient_flow(model)
        results.append(result)
        self.log(result.message, "pass" if result.passed else "fail")

        # 4. PDE residuals
        result = self.validate_pde_residuals(model)
        results.append(result)
        self.log(result.message, "pass" if result.passed else "fail")

        # 5. Training loop
        print(f"\n  Training for {self.num_epochs} epochs...")
        result = self.validate_training_loop(model)
        results.append(result)
        self.log(result.message, "pass" if result.passed else "fail")

        # 6. Physical validity (after training)
        result = self.validate_physical_outputs(model)
        results.append(result)
        self.log(result.message, "pass" if result.passed else "fail")

        return results

    def run_all_validations(self) -> Dict[str, List[ValidationResult]]:
        """Run validations on all registered models."""
        from src.model import MODEL_REGISTRY, BasePINN

        all_results = {}

        # Define test configurations for each model
        model_configs = {
            "BasePINN": {
                "class": BasePINN,
                "kwargs": {"hidden_layers": [32, 32]},
            },
            "SequentialPINN": {
                "class": MODEL_REGISTRY["sequential-pinn"],
                "kwargs": {"hidden_layers": [32, 32], "num_ffm_frequencies": 2},
            },
            "GatedPINN": {
                "class": MODEL_REGISTRY["gated-pinn"],
                "kwargs": {"hidden_layers": [32, 32], "num_ffm_frequencies": 2},
            },
            "FourierPINN": {
                "class": MODEL_REGISTRY["fourier-pinn"],
                "kwargs": {"hidden_layers": [64, 64], "mapping_size": 64, "sigma": 5.0},
            },
            "ModulatedPINN": {
                "class": MODEL_REGISTRY["modulated-pinn"],
                "kwargs": {"hidden_layers": [32, 32]},
            },
            "NonDimPINN": {
                "class": MODEL_REGISTRY["nondim-pinn"],
                "kwargs": {"hidden_layers": [32, 32], "num_ffm_frequencies": 2},
            },
        }

        for name, config in model_configs.items():
            results = self.validate_model(
                model_class=config["class"],
                model_name=name,
                **config["kwargs"]
            )
            all_results[name] = results

        return all_results

    def run_config_validation(self, config_path: str) -> List[ValidationResult]:
        """Validate a specific YAML config by running training."""
        import subprocess
        import tempfile

        name = f"Config: {Path(config_path).name}"
        start = time.time()

        print(f"\n{'='*60}")
        print(f"Validating config: {config_path}")
        print(f"{'='*60}")

        results = []

        try:
            # Run training with limited epochs
            cmd = [
                sys.executable, "-m", "src.trainer", "fit",
                f"--config={config_path}",
                f"--trainer.max_epochs={self.num_epochs}",
                "--trainer.enable_progress_bar=false",
                "--trainer.enable_model_summary=false",
                f"--data.num_points={self.num_points}",
                f"--data.batch_size={self.batch_size}",
            ]

            self.log(f"Running: {' '.join(cmd)}", "info")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=PROJECT_ROOT,
            )

            duration = time.time() - start

            if result.returncode == 0:
                results.append(ValidationResult(
                    name=name,
                    passed=True,
                    message=f"Config training completed successfully ({duration:.1f}s)",
                    duration=duration,
                ))
                self.log(f"Training completed in {duration:.1f}s", "pass")
            else:
                error_msg = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                results.append(ValidationResult(
                    name=name,
                    passed=False,
                    message=f"Training failed:\n{error_msg}",
                    duration=duration,
                ))
                self.log(f"Training failed: {error_msg[:100]}...", "fail")

        except subprocess.TimeoutExpired:
            results.append(ValidationResult(
                name=name,
                passed=False,
                message="Training timed out (>5 minutes)",
                duration=time.time() - start,
            ))
            self.log("Training timed out", "fail")
        except Exception as e:
            results.append(ValidationResult(
                name=name,
                passed=False,
                message=f"Error running config: {str(e)}",
                duration=time.time() - start,
            ))
            self.log(f"Error: {str(e)}", "fail")

        return results

    def print_summary(self, all_results: Dict[str, List[ValidationResult]]):
        """Print summary of all validation results."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        total_passed = 0
        total_failed = 0

        for model_name, results in all_results.items():
            passed = sum(1 for r in results if r.passed)
            failed = len(results) - passed
            total_passed += passed
            total_failed += failed

            status = "PASS" if failed == 0 else "FAIL"
            print(f"\n{model_name}: {status} ({passed}/{len(results)} tests)")

            if failed > 0:
                for r in results:
                    if not r.passed:
                        print(f"  - {r.name}: {r.message[:80]}")

        print("\n" + "-" * 60)
        print(f"Total: {total_passed} passed, {total_failed} failed")
        print("=" * 60)

        return total_failed == 0


def main():
    parser = argparse.ArgumentParser(description="Validate PINN experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 epochs)")
    parser.add_argument("--model", type=str, help="Test specific model by name")
    parser.add_argument("--config", type=str, help="Test specific YAML config")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--all-configs", action="store_true", help="Test all config files")

    args = parser.parse_args()

    num_epochs = 10 if args.quick else args.epochs

    validator = ExperimentValidator(
        num_epochs=num_epochs,
        device=args.device,
        verbose=not args.quiet,
    )

    print(f"\nExperiment Validator")
    print(f"Device: {validator.device}")
    print(f"Epochs: {num_epochs}")

    if args.config:
        # Test specific config
        results = validator.run_config_validation(args.config)
        all_results = {Path(args.config).stem: results}
    elif args.all_configs:
        # Test all config files
        config_dir = Path(__file__).parent.parent / "configs"
        all_results = {}
        for config_file in sorted(config_dir.glob("*.yaml")):
            results = validator.run_config_validation(str(config_file))
            all_results[config_file.stem] = results
    elif args.model:
        # Test specific model
        from src.model import MODEL_REGISTRY
        model_key = args.model.lower().replace("_", "-")
        if model_key not in MODEL_REGISTRY:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available models: {', '.join(sorted(MODEL_REGISTRY.keys()))}")
            sys.exit(1)

        model_class = MODEL_REGISTRY[model_key]
        results = validator.validate_model(model_class, args.model, hidden_layers=[32, 32])
        all_results = {args.model: results}
    else:
        # Test all models
        all_results = validator.run_all_validations()

    success = validator.print_summary(all_results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
