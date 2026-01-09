# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import warp as wp
import warp.fem as fem
from warp.fem import Field, Sample, Domain, integrand, normal


from vomp.fem.simulations.compress_two_cubes import build_cube_geo, clamp_bottom_face
from vomp.fem.fem_examples.mfem.softbody_sim import ClassicFEM


@fem.integrand
def volume_integrand(s: Sample, domain: Domain):
    """Integrand for computing volume."""
    return 1.0


def _compute_volume_change(sim: ClassicFEM) -> float:
    initial_volume = 1.0

    # Compute deformed volume using the Jacobian of the deformation
    @fem.integrand
    def deformed_volume_integrand(s: Sample, domain: Domain, u_cur: Field):
        F = fem.grad(u_cur, s) + wp.identity(n=3, dtype=float)
        J = wp.determinant(F)
        return J

    # Integrate over the domain to get the deformed volume
    deformed_volume = fem.integrate(
        deformed_volume_integrand,
        fields={"u_cur": sim.u_field},
        quadrature=sim.vel_quadrature,
        output_dtype=float,
    )

    # Return relative volume change
    return (float(deformed_volume) - initial_volume) / initial_volume


def _simulate_cube(
    young: float,
    poisson: float,
    density: float,
    *,
    resolution: int = 10,
    force: float = 3e3,
    n_frames: int = 30,
) -> Tuple[float, float]:
    """Runs a quasi-quasistatic vertical compression test on a *single* cube.

    Parameters
    ----------
    young, poisson, density : material parameters
    resolution              : grid resolution along one axis
    force                   : downward compressive force (N)
    n_frames                : solver frames (iterations)

    Returns
    -------
    (Final potential energy, Volume change ratio)
    """

    geo = build_cube_geo(resolution, offset=(0.0, 0.0))

    parser = argparse.ArgumentParser(add_help=False)
    ClassicFEM.add_parser_arguments(parser)

    args = parser.parse_args([])

    args.young_modulus = young
    args.poisson_ratio = poisson
    args.density = density

    args.quasi_quasistatic = True
    args.gravity = 0.0
    args.n_frames = n_frames

    sim = ClassicFEM(geo, None, args)
    sim.init_displacement_space()
    sim.init_strain_spaces()

    sim.set_boundary_condition(boundary_projector_form=clamp_bottom_face)

    top_center = wp.vec3(0.5, 1.0, 0.5)
    sim.forces.count = 1
    sim.forces.centers = wp.array([top_center], dtype=wp.vec3)
    sim.forces.radii = wp.array([0.6], dtype=float)
    sim.forces.forces = wp.array([wp.vec3(0.0, -force, 0.0)], dtype=wp.vec3)
    sim.update_force_weight()

    sim.init_constant_forms()
    sim.project_constant_forms()

    for _ in range(n_frames):
        sim.run_frame()

    E, _ = sim.evaluate_energy()
    volume_change = _compute_volume_change(sim)
    return float(E), volume_change


def _simulate_cube_stretch(
    young, poisson, density, *, resolution=10, force=3e3, n_frames=30
):
    """Pull right face of cube in +X with given force.

    Returns
    -------
    (Final potential energy, Volume change ratio)
    """
    geo = build_cube_geo(resolution, offset=(0.0, 0.0))

    parser_tmp = argparse.ArgumentParser(add_help=False)
    ClassicFEM.add_parser_arguments(parser_tmp)
    args_tmp = parser_tmp.parse_args([])
    args_tmp.young_modulus = young
    args_tmp.poisson_ratio = poisson
    args_tmp.density = density
    args_tmp.quasi_quasistatic = True
    args_tmp.gravity = 0.0
    args_tmp.n_frames = n_frames

    sim = ClassicFEM(geo, None, args_tmp)
    sim.init_displacement_space()
    sim.init_strain_spaces()

    @integrand
    def clamp_left_face(s: Sample, domain: Domain, u: Field, v: Field):
        nor = normal(domain, s)
        clamped = wp.where(nor[0] < 0.0, 1.0, 0.0)
        return wp.dot(u(s), v(s)) * clamped

    sim.set_boundary_condition(boundary_projector_form=clamp_left_face)

    right_center = wp.vec3(1.0, 0.5, 0.5)
    sim.forces.count = 1
    sim.forces.centers = wp.array([right_center], dtype=wp.vec3)
    sim.forces.radii = wp.array([0.6], dtype=float)
    sim.forces.forces = wp.array([wp.vec3(force, 0.0, 0.0)], dtype=wp.vec3)
    sim.update_force_weight()

    sim.init_constant_forms()
    sim.project_constant_forms()

    for _ in range(n_frames):
        sim.run_frame()

    E, _ = sim.evaluate_energy()
    volume_change = _compute_volume_change(sim)
    return float(E), volume_change


def main():
    wp.init()

    parser = argparse.ArgumentParser(
        description="Material sensitivity test – PE and volumetric displacement under compression vs. (ρ, ν, E)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--resolution", type=int, default=10, help="Cube grid resolution"
    )
    parser.add_argument(
        "--force", type=float, default=3e3, help="Compressive force (N)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Quasi-static solver frames (per simulation)",
    )
    parser.add_argument(
        "--rel_changes",
        type=float,
        nargs="*",
        default=[0.01, 0.05, 0.10, 0.20, 0.25, 0.30],
        help="Relative changes (proportions): 0.01 → 1 %, 0.05 → 5 %, etc.",
    )
    parser.add_argument(
        "--triplets",
        type=float,
        nargs="*",
        default=[],
        help="Baseline material triplets listed as (E, ν, ρ) …",
    )
    parser.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a very small test grid (≈1–2 triplets) to validate the setup",
    )
    parser.add_argument(
        "--mode",
        choices=["compress", "stretch", "both"],
        default="compress",
        help="Which loading scenario to run: compression, stretch, or both",
    )
    default_out = Path(__file__).resolve().parent
    parser.add_argument(
        "--out_dir", type=str, default=str(default_out), help="Where to save results"
    )
    args = parser.parse_args()

    out_dir_path = Path(args.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir_path)

    if args.triplets:
        if len(args.triplets) % 3 != 0:
            sys.exit("--triplets must contain a multiple of 3 values (E, ν, ρ …)")
        triplets: List[Tuple[float, float, float]] = [
            tuple(args.triplets[i : i + 3]) for i in range(0, len(args.triplets), 3)
        ]
    else:

        from itertools import product

        if args.quick:

            E_grid = [1e5, 1e6]
            nu_grid = [0.30]
            rho_grid = [1000.0]
        else:

            E_grid = [1e8, 1e6, 1e5]
            nu_grid = [0.3, 0.35, 0.45]
            rho_grid = [1000.0, 4000.0, 6000.0]

        triplets = list(product(E_grid, nu_grid, rho_grid))

    # Generate all material configurations to simulate
    all_configs = []
    for E0, nu0, rho0 in triplets:
        # Add baseline
        all_configs.append((E0, nu0, rho0))

        # Add variations
        for rel in args.rel_changes:
            # Density variations
            rho_new = rho0 * (1.0 + rel)
            all_configs.append((E0, nu0, rho_new))

            # Poisson ratio variations
            nu_new = nu0 * (1.0 + rel)
            if nu_new < 0.499:
                all_configs.append((E0, nu_new, rho0))

            # Young's modulus variations (using exponential scaling)
            E_new = E0 * math.exp(rel)
            all_configs.append((E_new, nu0, rho0))

    # Remove duplicates
    all_configs = list(set(all_configs))

    total_est = len(all_configs)
    print(f"Running {total_est} unique simulations...")

    _pe_cache = {}
    _vol_cache = {}

    pbar = tqdm(total=total_est, desc="Simulations", unit="sim")

    def cached_simulate(E, nu, rho, scenario):
        key = (E, nu, rho, args.resolution, args.force, args.frames, scenario)
        if key not in _pe_cache:
            if scenario == "compress":
                pe, vol_change = _simulate_cube(
                    E,
                    nu,
                    rho,
                    resolution=args.resolution,
                    force=args.force,
                    n_frames=args.frames,
                )
            else:
                pe, vol_change = _simulate_cube_stretch(
                    E,
                    nu,
                    rho,
                    resolution=args.resolution,
                    force=args.force,
                    n_frames=args.frames,
                )
            _pe_cache[key] = pe
            _vol_cache[key] = vol_change
            pbar.update(1)
        return _pe_cache[key], _vol_cache[key]

    scenarios = [args.mode] if args.mode != "both" else ["compress", "stretch"]

    for scenario in scenarios:

        scenario_dir = out_dir_path / scenario
        scenario_dir.mkdir(exist_ok=True)

        # First, run all simulations and store results
        sim_results = []
        for E, nu, rho in all_configs:
            PE, VOL = cached_simulate(E, nu, rho, scenario)
            sim_results.append({"E": E, "nu": nu, "rho": rho, "PE": PE, "VOL": VOL})

        # Save raw simulation results
        raw_df = pd.DataFrame(sim_results)
        # raw_df.to_csv(scenario_dir / "raw_simulation_results.csv", index=False)

        # Now compute pairwise relative differences
        records = []

        for i, sim1 in enumerate(sim_results):
            for j, sim2 in enumerate(sim_results):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue

                # Compute relative changes in parameters
                rel_E = (sim2["E"] - sim1["E"]) / sim1["E"]
                rel_nu = (
                    (sim2["nu"] - sim1["nu"]) / sim1["nu"] if sim1["nu"] != 0 else 0
                )
                rel_rho = (
                    (sim2["rho"] - sim1["rho"]) / sim1["rho"] if sim1["rho"] != 0 else 0
                )

                # Compute relative changes in outputs
                rel_PE = (
                    (sim2["PE"] - sim1["PE"]) / sim1["PE"] if sim1["PE"] != 0 else 0
                )
                rel_VOL = (
                    (sim2["VOL"] - sim1["VOL"]) / sim1["VOL"] if sim1["VOL"] != 0 else 0
                )

                # Determine which parameter changed (if only one)
                param_changed = None
                param_rel_change = 0

                # Check if only one parameter changed significantly (threshold 1e-6)
                changes = []
                if abs(rel_E) > 1e-6:
                    changes.append(("E", rel_E))
                if abs(rel_nu) > 1e-6:
                    changes.append(("nu", rel_nu))
                if abs(rel_rho) > 1e-6:
                    changes.append(("rho", rel_rho))

                if len(changes) == 1:
                    param_changed, param_rel_change = changes[0]

                records.append(
                    {
                        "sim1_idx": i,
                        "sim2_idx": j,
                        "E1": sim1["E"],
                        "nu1": sim1["nu"],
                        "rho1": sim1["rho"],
                        "E2": sim2["E"],
                        "nu2": sim2["nu"],
                        "rho2": sim2["rho"],
                        "rel_E": rel_E,
                        "rel_nu": rel_nu,
                        "rel_rho": rel_rho,
                        "param_changed": param_changed,
                        "param_rel_change": param_rel_change,
                        "rel_PE": rel_PE,
                        "rel_VOL": rel_VOL,
                        "abs_rel_PE": abs(rel_PE),
                        "abs_rel_VOL": abs(rel_VOL),
                    }
                )

        # Save all pairwise comparisons
        df = pd.DataFrame(records)
        # df.to_csv(scenario_dir / "pairwise_comparisons.csv", index=False)

        # Create simplified summary statistics for specified relative changes only
        summary_records = []
        for param in ["E", "nu", "rho"]:
            param_df = df[df["param_changed"] == param]

            if len(param_df) == 0:
                continue

            # For each specified relative change
            for rel_change in args.rel_changes:
                # Find data points close to this relative change (within 1% tolerance)
                tolerance = 0.01
                matching_data = param_df[
                    (param_df["param_rel_change"] >= rel_change - tolerance)
                    & (param_df["param_rel_change"] <= rel_change + tolerance)
                ]

                if len(matching_data) > 0:
                    summary_records.append(
                        {
                            "parameter": param,
                            "relative_change": rel_change,
                            "PE_mean": matching_data["rel_PE"].mean(),
                            "PE_std": matching_data["rel_PE"].std(),
                            "VOL_mean": matching_data["rel_VOL"].mean(),
                            "VOL_std": matching_data["rel_VOL"].std(),
                        }
                    )

        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(scenario_dir / "sensitivity_summary.csv", index=False)

        # For plotting, we still need the binned data
        plot_summary_records = []
        for param in ["E", "nu", "rho"]:
            param_df = df[df["param_changed"] == param]

            if len(param_df) == 0:
                continue

            # Group by approximate parameter change bins for plotting
            bins = np.arange(-0.35, 0.36, 0.05)
            param_df["param_bin"] = pd.cut(param_df["param_rel_change"], bins)

            for bin_val in param_df["param_bin"].unique():
                if pd.isna(bin_val):
                    continue

                bin_data = param_df[param_df["param_bin"] == bin_val]
                if len(bin_data) == 0:
                    continue

                plot_summary_records.append(
                    {
                        "parameter": param,
                        "param_change_bin": str(bin_val),
                        "param_change_mean": bin_data["param_rel_change"].mean(),
                        "n_samples": len(bin_data),
                        "PE_rel_mean": bin_data["rel_PE"].mean(),
                        "PE_rel_std": bin_data["rel_PE"].std(),
                        "VOL_rel_mean": bin_data["rel_VOL"].mean(),
                        "VOL_rel_std": bin_data["rel_VOL"].std(),
                    }
                )

        plot_summary_df = pd.DataFrame(plot_summary_records)

        # Create sensitivity plots from the plot summary data
        for param in ["E", "nu", "rho"]:
            param_summary = plot_summary_df[
                plot_summary_df["parameter"] == param
            ].copy()

            if len(param_summary) == 0:
                continue

            # Sort by mean parameter change
            param_summary = param_summary.sort_values("param_change_mean")

            x = param_summary["param_change_mean"].values
            y_pe = param_summary["PE_rel_mean"].values
            y_pe_std = param_summary["PE_rel_std"].values
            y_vol = param_summary["VOL_rel_mean"].values
            y_vol_std = param_summary["VOL_rel_std"].values

            # Configure matplotlib to use Helvetica font
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
            plt.rcParams["font.size"] = 5

            # Plot PE sensitivity
            fig, ax = plt.subplots(figsize=(2.1, 1.8))

            # Remove title
            # plt.title(f'PE vs Δ{param}')

            # Set axis labels with Delta notation
            param_symbol = {"E": "E", "nu": "ν", "rho": "ρ"}[param]
            ax.set_xlabel(
                f"$\\frac{{\\Delta {param_symbol}}}{{{param_symbol}}}$", fontsize=5
            )

            # Determine scale factor for y-axis
            y_max_abs = max(
                abs(y_pe.max()),
                abs(y_pe.min()),
                abs((y_pe + y_pe_std).max()),
                abs((y_pe - y_pe_std).min()),
            )
            if y_max_abs == 0:
                y_scale_factor = 1
                y_scale_label = ""
            elif y_max_abs < 0.01:
                # Scale up to show in reasonable range
                exponent = int(np.floor(np.log10(y_max_abs)))
                y_scale_factor = 10 ** (-exponent)
                y_scale_label = f"×10$^{{{exponent}}}$"
            elif y_max_abs > 10:
                # Scale down
                exponent = int(np.floor(np.log10(y_max_abs)))
                y_scale_factor = 10 ** (-exponent)
                y_scale_label = f"×10$^{{{exponent}}}$"
            else:
                y_scale_factor = 1
                y_scale_label = ""

            # Scale the data
            y_pe_scaled = y_pe * y_scale_factor
            y_pe_std_scaled = y_pe_std * y_scale_factor

            # Plot with confidence bands using light solid colors
            (line,) = ax.plot(
                x, y_pe_scaled, "-o", color="C0", markersize=4, linewidth=1.5
            )
            # Use a light blue color for the confidence band (no transparency)
            ax.fill_between(
                x,
                y_pe_scaled - y_pe_std_scaled,
                y_pe_scaled + y_pe_std_scaled,
                color="#CCE5FF",  # Light blue, solid color
                edgecolor="none",
                zorder=0,
            )  # Put behind the line

            # Remove grid and dashed lines at 0
            ax.grid(False)

            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Make remaining borders thicker
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)

            # Set y-axis label horizontally at the top, starting from the y-axis position
            if y_scale_label:
                y_label_text = f"$\\frac{{\\Delta PE}}{{PE}}$ ({y_scale_label})"
            else:
                y_label_text = "$\\frac{\\Delta PE}{PE}$"
            ax.text(
                0,
                1.02,
                y_label_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=5,
            )

            # Add some padding between data and borders
            ax.margins(x=0.05, y=0.1)

            # Set tick label font size
            ax.tick_params(axis="both", which="major", labelsize=5)

            plt.tight_layout(pad=0)
            plt.savefig(
                scenario_dir / f"sensitivity_{param}_PE.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            # Plot Volume sensitivity
            fig, ax = plt.subplots(figsize=(2.1, 1.8))

            # Remove title
            # plt.title(f'Volume Change vs Δ{param}')

            # Set axis labels with Delta notation
            ax.set_xlabel(
                f"$\\frac{{\\Delta {param_symbol}}}{{{param_symbol}}}$", fontsize=5
            )

            # Determine scale factor for y-axis
            y_max_abs = max(
                abs(y_vol.max()),
                abs(y_vol.min()),
                abs((y_vol + y_vol_std).max()),
                abs((y_vol - y_vol_std).min()),
            )
            if y_max_abs == 0:
                y_scale_factor = 1
                y_scale_label = ""
            elif y_max_abs < 0.01:
                # Scale up to show in reasonable range
                exponent = int(np.floor(np.log10(y_max_abs)))
                y_scale_factor = 10 ** (-exponent)
                y_scale_label = f"×10$^{{{exponent}}}$"
            elif y_max_abs > 10:
                # Scale down
                exponent = int(np.floor(np.log10(y_max_abs)))
                y_scale_factor = 10 ** (-exponent)
                y_scale_label = f"×10$^{{{exponent}}}$"
            else:
                y_scale_factor = 1
                y_scale_label = ""

            # Scale the data
            y_vol_scaled = y_vol * y_scale_factor
            y_vol_std_scaled = y_vol_std * y_scale_factor

            # Plot with confidence bands using light solid colors
            (line,) = ax.plot(
                x, y_vol_scaled, "-o", color="C1", markersize=4, linewidth=1.5
            )
            # Use a light orange color for the confidence band (no transparency)
            ax.fill_between(
                x,
                y_vol_scaled - y_vol_std_scaled,
                y_vol_scaled + y_vol_std_scaled,
                color="#FFE5CC",  # Light orange, solid color
                edgecolor="none",
                zorder=0,
            )  # Put behind the line

            # Remove grid and dashed lines at 0
            ax.grid(False)

            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Make remaining borders thicker
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)

            # Set y-axis label horizontally at the top, starting from the y-axis position
            if y_scale_label:
                y_label_text = f"$\\frac{{\\Delta V}}{{V}}$ ({y_scale_label})"
            else:
                y_label_text = "$\\frac{\\Delta V}{V}$"
            ax.text(
                0,
                1.02,
                y_label_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=5,
            )

            # Add some padding between data and borders
            ax.margins(x=0.05, y=0.1)

            # Set tick label font size
            ax.tick_params(axis="both", which="major", labelsize=5)

            plt.tight_layout(pad=0)
            plt.savefig(
                scenario_dir / f"sensitivity_{param}_VOL.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

    pbar.close()

    print("\nDone.")
    print(f"Results saved to: {args.out_dir}")
    print(f"Total unique simulations run: {len(all_configs)}")
    print(f"Total pairwise comparisons: {len(records)}")


if __name__ == "__main__":
    main()
