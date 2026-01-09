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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("datasets/latent_space/materials.csv")

print("Data shape:", df.shape)
print("\nColumn names:", df.columns.tolist())


for col in ["youngs_modulus", "poisson_ratio", "density"]:
    print(f"\n{col}:")
    print(f"  Min: {df[col].min():.2e}")
    print(f"  Max: {df[col].max():.2e}")
    print(f"  Mean: {df[col].mean():.2e}")
    print(f"  Median: {df[col].median():.2e}")
    print(f"  Std: {df[col].std():.2e}")

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(
        f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)"
    )

    if col in ["youngs_modulus", "density"]:
        log_vals = np.log10(df[col])
        print(f"  Log10 range: [{log_vals.min():.2f}, {log_vals.max():.2f}]")
        print(
            f"  Log10 span: {log_vals.max() - log_vals.min():.2f} orders of magnitude"
        )


print("\n\nMaterials with extreme Young's modulus (< 1e7 Pa):")
low_E = df[df["youngs_modulus"] < 1e7]
if len(low_E) > 0:
    material_counts = low_E["material_name"].value_counts().head(10)
    for mat, count in material_counts.items():
        print(f"  {mat}: {count} samples")

print("\n\nMaterials with extreme density (< 100 kg/m³):")
low_rho = df[df["density"] < 100]
if len(low_rho) > 0:
    material_counts = low_rho["material_name"].value_counts().head(10)
    for mat, count in material_counts.items():
        print(f"  {mat}: {count} samples")


print("\n\nPercentile analysis:")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for col in ["youngs_modulus", "poisson_ratio", "density"]:
    print(f"\n{col} percentiles:")
    for p in percentiles:
        val = df[col].quantile(p / 100)
        print(f"  {p}%: {val:.2e}")


print("\n\nCreating filtered dataset...")

filtered_df = df[
    (df["youngs_modulus"] >= 1e5)
    & (df["youngs_modulus"] <= 1e12)
    & (df["density"] >= 100)
    & (df["density"] <= 20000)
    & (df["poisson_ratio"] >= 0.0)
    & (df["poisson_ratio"] <= 0.49)
]

print(f"Original size: {len(df)}")
print(f"Filtered size: {len(filtered_df)}")
print(
    f"Removed: {len(df) - len(filtered_df)} ({(len(df) - len(filtered_df))/len(df)*100:.1f}%)"
)


print("\nRanges in filtered dataset (only Poisson ratio filtering):")
for col in ["youngs_modulus", "poisson_ratio", "density"]:
    print(f"\n{col}:")
    print(f"  Min: {filtered_df[col].min():.2e}")
    print(f"  Max: {filtered_df[col].max():.2e}")
    print(f"  Range span: {filtered_df[col].max() - filtered_df[col].min():.2e}")

    if col in ["youngs_modulus", "density"]:
        log_min = np.log10(filtered_df[col].min())
        log_max = np.log10(filtered_df[col].max())
        print(f"  Log10 range: [{log_min:.2f}, {log_max:.2f}]")
        print(f"  Orders of magnitude: {log_max - log_min:.2f}")

filtered_df.to_csv("datasets/latent_space/materials_filtered.csv", index=False)
print("\nSaved filtered dataset to materials_filtered.csv")
