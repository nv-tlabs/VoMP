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


import json
import torch
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from matplotlib.ticker import MultipleLocator, FixedLocator
import warnings

warnings.filterwarnings("ignore")


def create_individual_histogram_plots():
    print("Loading materials data...")

    df = pd.read_csv("../../datasets/latent_space/materials_filtered.csv")

    E_values = df["youngs_modulus"].values
    nu_values = df["poisson_ratio"].values
    rho_values = df["density"].values

    print(f"Loaded {len(df)} materials")
    print(f"Young's Modulus range: {np.min(E_values):.2e} - {np.max(E_values):.2e} Pa")
    print(f"Poisson's Ratio range: {np.min(nu_values):.3f} - {np.max(nu_values):.3f}")
    print(f"Density range: {np.min(rho_values):.1f} - {np.max(rho_values):.1f} kg/m³")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcdefaults()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    fontsize = 13
    plt.rcParams["font.size"] = fontsize

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_facecolor("#f2f2f2")
    fig.patch.set_facecolor("white")

    log_E = np.log10(E_values)

    bins = np.linspace(log_E.min(), log_E.max(), 51)
    counts, bin_edges = np.histogram(log_E, bins=bins)
    total = len(log_E)
    probabilities = counts / total

    width = bins[1] - bins[0]
    ax.bar(
        bin_edges[:-1],
        probabilities,
        width=width,
        color="#8AB6D6",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    kde_x = np.linspace(log_E.min(), log_E.max(), 1000)
    kde = sns.kdeplot(
        log_E,
        ax=ax,
        color="#5A9BCF",
        linewidth=1.5,
        alpha=0.7,
        clip=(log_E.min(), log_E.max()),
    )

    line = kde.get_lines()[-1]
    kde_y = line.get_ydata()
    if max(kde_y) > 0:
        scale_factor = max(probabilities) / max(kde_y)
        line.set_ydata(kde_y * scale_factor)

    ax.set_xlim(log_E.min(), log_E.max())
    ax.set_xlabel("Young's Modulus (log Pa)", fontsize=fontsize)
    ax.set_ylabel("")

    y_max = max(probabilities) * 1.1 if max(probabilities) > 0 else 1.0
    ax.set_ylim(0, y_max)

    ytick_values = np.arange(0.04, y_max, 0.04)
    ax.yaxis.set_major_locator(FixedLocator(ytick_values))
    ax.set_yticklabels([f"{y:.2f}" for y in ytick_values])

    ax.yaxis.set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.tick_params(axis="y", which="both", left=True, colors="black", length=4, width=1)

    ax.text(
        -0.08,
        1.01,
        "Probability",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="y", linestyle="-", color="white", linewidth=1.0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = output_dir / "youngs_modulus_histogram.pdf"
    plt.savefig(output_path, dpi=300, format="pdf", bbox_inches="tight")
    print(f"Saved Young's Modulus histogram to {output_path}")
    plt.close()

    plt.rcdefaults()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    fontsize = 13
    plt.rcParams["font.size"] = fontsize

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_facecolor("#f2f2f2")
    fig.patch.set_facecolor("white")

    bins = np.linspace(nu_values.min(), nu_values.max(), 51)
    counts, bin_edges = np.histogram(nu_values, bins=bins)
    total = len(nu_values)
    probabilities = counts / total

    width = bins[1] - bins[0]
    ax.bar(
        bin_edges[:-1],
        probabilities,
        width=width,
        color="#8AB6D6",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    kde_x = np.linspace(nu_values.min(), nu_values.max(), 1000)
    kde = sns.kdeplot(
        nu_values,
        ax=ax,
        color="#5A9BCF",
        linewidth=1.5,
        alpha=0.7,
        clip=(nu_values.min(), nu_values.max()),
    )

    line = kde.get_lines()[-1]
    kde_y = line.get_ydata()
    if max(kde_y) > 0:
        scale_factor = max(probabilities) / max(kde_y)
        line.set_ydata(kde_y * scale_factor)

    ax.set_xlim(nu_values.min(), nu_values.max())
    ax.set_xlabel("Poisson's Ratio", fontsize=fontsize)
    ax.set_ylabel("")

    y_max = max(probabilities) * 1.1 if max(probabilities) > 0 else 1.0
    ax.set_ylim(0, y_max)

    ytick_values = np.arange(0.02, y_max, 0.02)
    ax.yaxis.set_major_locator(FixedLocator(ytick_values))
    ax.set_yticklabels([f"{y:.2f}" for y in ytick_values])

    ax.yaxis.set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.tick_params(axis="y", which="both", left=True, colors="black", length=4, width=1)

    ax.text(
        -0.08,
        1.01,
        "Probability",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="y", linestyle="-", color="white", linewidth=1.0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = output_dir / "poisson_ratio_histogram.pdf"
    plt.savefig(output_path, dpi=300, format="pdf", bbox_inches="tight")
    print(f"Saved Poisson's Ratio histogram to {output_path}")
    plt.close()

    plt.rcdefaults()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    fontsize = 13
    plt.rcParams["font.size"] = fontsize

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_facecolor("#f2f2f2")
    fig.patch.set_facecolor("white")

    log_rho = np.log10(rho_values)

    bins = np.linspace(log_rho.min(), log_rho.max(), 51)
    counts, bin_edges = np.histogram(log_rho, bins=bins)
    total = len(log_rho)
    probabilities = counts / total

    width = bins[1] - bins[0]
    ax.bar(
        bin_edges[:-1],
        probabilities,
        width=width,
        color="#8AB6D6",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    kde_x = np.linspace(log_rho.min(), log_rho.max(), 1000)
    kde = sns.kdeplot(
        log_rho,
        ax=ax,
        color="#5A9BCF",
        linewidth=1.5,
        alpha=0.7,
        clip=(log_rho.min(), log_rho.max()),
    )

    line = kde.get_lines()[-1]
    kde_y = line.get_ydata()
    if max(kde_y) > 0:
        scale_factor = max(probabilities) / max(kde_y)
        line.set_ydata(kde_y * scale_factor)

    ax.set_xlim(log_rho.min(), log_rho.max())
    ax.set_xlabel("Density (log kg/m³)", fontsize=fontsize)
    ax.set_ylabel("")

    y_max = max(probabilities) * 1.1 if max(probabilities) > 0 else 1.0
    ax.set_ylim(0, y_max)

    ytick_values = np.arange(0.04, y_max, 0.04)
    ax.yaxis.set_major_locator(FixedLocator(ytick_values))
    ax.set_yticklabels([f"{y:.2f}" for y in ytick_values])

    ax.yaxis.set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.tick_params(axis="y", which="both", left=True, colors="black", length=4, width=1)

    ax.text(
        -0.08,
        1.01,
        "Probability",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="y", linestyle="-", color="white", linewidth=1.0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = output_dir / "density_histogram.pdf"
    plt.savefig(output_path, dpi=300, format="pdf", bbox_inches="tight")
    print(f"Saved Density histogram to {output_path}")
    plt.close()

    return E_values, nu_values, rho_values


if __name__ == "__main__":
    print("Creating individual material property histograms...")
    create_individual_histogram_plots()

    print("\nHistogram generation complete!")
