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
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROPERTIES = ("youngs_modulus", "poisson_ratio", "density")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build MatVAE training and validation CSVs from the material triplet dataset."
        )
    )
    parser.add_argument(
        "--latent_dir", type=Path, default=Path("datasets/latent_space")
    )
    return parser.parse_args()


def triplet_array(frame):
    values = frame.loc[:, PROPERTIES].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Material properties contain non-finite values")
    if np.any(values[:, 0] <= 0) or np.any(values[:, 2] <= 0):
        raise ValueError("Young's modulus and density must be positive")
    if np.any(values[:, 1] <= -1.0) or np.any(values[:, 1] > 0.5):
        raise ValueError("Poisson's ratio must be in (-1, 0.5]")
    return values


def byte_keys(values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    return [row.tobytes() for row in values]


def normalized(values, bounds):
    values = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    E_min, E_max, nu_min, nu_max, rho_min, rho_max = bounds
    E_min = values.new_tensor(E_min)
    E_max = values.new_tensor(E_max)
    rho_min = values.new_tensor(rho_min)
    rho_max = values.new_tensor(rho_max)
    E = (torch.log10(values[:, 0]) - torch.log10(E_min)) / (
        torch.log10(E_max) - torch.log10(E_min)
    )
    nu = (values[:, 1] - nu_min) / (nu_max - nu_min)
    rho = (torch.log10(values[:, 2]) - torch.log10(rho_min)) / (
        torch.log10(rho_max) - torch.log10(rho_min)
    )
    return torch.stack((E, nu, rho), dim=-1).numpy()


def normalization_bounds(*arrays):
    values = np.concatenate(arrays, axis=0)
    return (
        float(values[:, 0].min()),
        float(values[:, 0].max()),
        float(values[:, 1].min()),
        float(values[:, 1].max()),
        float(values[:, 2].min()),
        float(values[:, 2].max()),
    )


def target_frame(targets, columns):
    records = []
    for values in targets:
        digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()[:16]
        row = {column: np.nan for column in columns}
        row.update(
            {
                "material_name": f"simready_target_{digest}",
                "youngs_modulus": float(values[0]),
                "poisson_ratio": float(values[1]),
                "density": float(values[2]),
                "record_id": f"simready_target_{digest}",
                "material_family": "SimReady exact target",
                "range_coupling": "exact downstream annotation",
                "sampling_coupling": "fixed_point",
                "sample_kind": "fixed_anchor",
                "source_t": np.nan,
                "weight_reason": "uniform downstream target coverage",
                "split": "train",
                "protected_train": True,
            }
        )
        records.extend([row.copy() for _ in range(18)])
    return pd.DataFrame.from_records(records)


def split_validation_records(source):
    """Hold out whole records, stratified across every non-singleton family."""
    record_table = (
        source.groupby("record_id", sort=True)
        .agg(
            rows=("record_id", "size"),
            material_family=("material_family", "first"),
            family_count=("material_family", "nunique"),
            sampling_coupling=("sampling_coupling", "first"),
            coupling_count=("sampling_coupling", "nunique"),
        )
        .reset_index()
    )
    if int(record_table[["family_count", "coupling_count"]].to_numpy().max()) != 1:
        raise ValueError("Each source record must have one family and one coupling")
    unsupported = set(record_table["sampling_coupling"]) - {
        "fixed_point",
        "single_property_interval",
        "source_indicated_correlated_shared_t",
        "conservative_shared_t_within_same_record",
        "conservative_shared_t_through_marginal_bounds",
    }
    if unsupported:
        raise ValueError(f"Unsupported sampling coupling(s): {sorted(unsupported)}")

    held_out = []
    for family, family_records in record_table.groupby("material_family", sort=True):
        if len(family_records) < 2:
            continue
        count = max(1, round(len(family_records) * 0.10))
        count = min(count, len(family_records) - 1)
        family_records = family_records.copy()
        family_records["_record_score"] = family_records["record_id"].map(
            lambda record: hashlib.sha256(
                f"42\0validation\0{family}\0{record}".encode()
            ).hexdigest()
        )
        held_out.extend(
            family_records.sort_values("_record_score").head(count)["record_id"]
        )

    held_out = set(held_out)
    swaps = []
    coupling_record_counts = record_table["sampling_coupling"].value_counts()
    required_couplings = set(coupling_record_counts.loc[lambda count: count >= 2].index)
    family_record_counts = record_table["material_family"].value_counts()
    family_row_counts = source["material_family"].value_counts()
    for missing in sorted(
        required_couplings
        - set(record_table.loc[record_table["record_id"].isin(held_out), "sampling_coupling"])
    ):
        held_table = record_table.loc[record_table["record_id"].isin(held_out)]
        held_coupling_counts = held_table["sampling_coupling"].value_counts()
        candidates = record_table.loc[
            ~record_table["record_id"].isin(held_out)
            & record_table["sampling_coupling"].eq(missing)
            & record_table["material_family"].map(family_record_counts).gt(1)
        ]
        pairs = []
        for candidate in candidates.itertuples(index=False):
            removable = held_table.loc[held_table["rows"].eq(candidate.rows)]
            for old in removable.itertuples(index=False):
                if held_coupling_counts[old.sampling_coupling] <= 1:
                    continue
                score = hashlib.sha256(
                    f"42\0coupling-swap\0{missing}\0{candidate.record_id}\0{old.record_id}".encode()
                ).hexdigest()
                pairs.append(
                    (
                        int(family_row_counts[old.material_family]),
                        score,
                        candidate.record_id,
                        old.record_id,
                    )
                )
        if not pairs:
            raise RuntimeError(
                f"Cannot cover {missing} without changing the held-out source-row count"
            )
        _, _, new_record, old_record = min(pairs)
        held_out.remove(old_record)
        held_out.add(new_record)
        swaps.append(f"{old_record} -> {new_record}")

    held_source = source.loc[source["record_id"].isin(held_out)].copy()
    train_source = source.loc[~source["record_id"].isin(held_out)].copy()
    if len(train_source) != 44_645:
        raise RuntimeError(
            f"Record holdout leaves {len(train_source)} source rows; "
            "expected 44645"
        )
    if set(held_source["record_id"]) & set(train_source["record_id"]):
        raise RuntimeError("A held-out record leaked into the training candidates")
    return train_source, held_source, held_out, swaps


def endpoint_indices(frame):
    endpoints = set()
    for column in (*PROPERTIES, "source_t"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            endpoints.add(values.idxmin())
            endpoints.add(values.idxmax())
    return endpoints


def sample_validation_record(frame):
    """Keep endpoints and fill the cap by farthest-point sampling along source_t."""
    if len(frame) <= 128:
        return frame.copy()
    ordered = frame.sort_values(["source_t", *PROPERTIES], na_position="last")
    positions = {index: position for position, index in enumerate(ordered.index)}
    selected = {positions[index] for index in endpoint_indices(frame)}
    if len(selected) > 128:
        raise RuntimeError("A validation record has too many distinct endpoints")

    source_t = pd.to_numeric(ordered["source_t"], errors="coerce").to_numpy()
    if np.isfinite(source_t).all() and source_t.max() > source_t.min():
        coordinate = (source_t - source_t.min()) / (source_t.max() - source_t.min())
    else:
        coordinate = np.linspace(0.0, 1.0, len(ordered))
    while len(selected) < 128:
        selected_array = np.fromiter(selected, dtype=np.int64)
        distance = np.min(
            np.abs(coordinate[:, None] - coordinate[selected_array][None, :]), axis=1
        )
        distance[selected_array] = -1.0
        selected.add(int(np.argmax(distance)))
    return ordered.iloc[sorted(selected)].copy()


def sample_validation(held_source):
    return pd.concat(
        [sample_validation_record(frame) for _, frame in held_source.groupby("record_id")],
        ignore_index=True,
    )


def validate_written(train_path, validation_path, targets, original_source, held_records):
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    train_values = triplet_array(train)
    validation_values = triplet_array(validation)
    if len(train) != 90_959:
        raise RuntimeError(
            f"Expected 90959 rows, found {len(train)}"
        )

    bounds = normalization_bounds(train_values)
    train_norm = normalized(train_values, bounds)
    validation_norm = normalized(validation_values, bounds)
    train_keys = byte_keys(train_norm)
    target_keys = set(byte_keys(normalized(targets, bounds)))
    protected_keys = set(
        key
        for key, protected in zip(
            train_keys,
            train["protected_train"].fillna(False).astype(bool).to_numpy(),
        )
        if protected
    )
    if protected_keys != target_keys:
        raise RuntimeError(
            f"Protected target recall is {len(protected_keys & target_keys)}/{len(target_keys)}"
        )
    if set(byte_keys(validation_norm)) & set(train_keys):
        raise RuntimeError("Training and validation contain normalized overlap")

    source_mask = ~train["protected_train"].fillna(False).astype(bool)
    train_source = train.loc[source_mask]
    train_records = set(train_source["record_id"].astype(str))
    validation_records = set(validation["record_id"].astype(str))
    held_records = {str(record) for record in held_records}
    if train_records & validation_records:
        raise RuntimeError("Validation contains a source record seen during training")
    if validation_records != held_records:
        raise RuntimeError("Every held-out record must be represented in validation")

    original_record_ids = original_source["record_id"].astype(str)
    expected_source = original_source.loc[~original_record_ids.isin(held_records)]
    held_source = original_source.loc[original_record_ids.isin(held_records)]
    if len(expected_source) != 44_645:
        raise RuntimeError("Unexpected number of non-validation source rows")
    if set(byte_keys(triplet_array(train_source))) != set(
        byte_keys(triplet_array(expected_source))
    ):
        raise RuntimeError("Training does not preserve every non-validation source triplet")
    if not set(byte_keys(validation_values)) <= set(byte_keys(triplet_array(held_source))):
        raise RuntimeError("Validation contains a triplet outside its held-out records")
    validation_sizes = validation.groupby("record_id").size()
    if int(validation_sizes.max()) > 128:
        raise RuntimeError("A validation record exceeds its row cap")
    for record, frame in held_source.groupby("record_id"):
        actual = validation.loc[validation["record_id"].eq(record)]
        endpoint_keys = set(byte_keys(triplet_array(frame.loc[list(endpoint_indices(frame))])))
        if not endpoint_keys <= set(byte_keys(triplet_array(actual))):
            raise RuntimeError(f"Validation omitted an endpoint for {record}")

    coupling_record_counts = original_source.groupby("sampling_coupling")[
        "record_id"
    ].nunique()
    required_couplings = set(coupling_record_counts.loc[lambda count: count >= 2].index)
    validation_couplings = set(validation["sampling_coupling"])
    if not required_couplings <= validation_couplings:
        raise RuntimeError("Validation does not cover every eligible sampling coupling")

    _, multiplicity = np.unique(train_norm, axis=0, return_counts=True)
    effective_size = float(len(train) ** 2 / np.square(multiplicity).sum())
    if int(multiplicity.max()) > 18:
        raise RuntimeError(f"Maximum triplet multiplicity is {multiplicity.max()}")
    counts = pd.Series(train_keys).value_counts()
    target_counts = counts.reindex(list(target_keys), fill_value=0)
    if not target_counts.eq(18).all():
        raise RuntimeError(
            "Every protected target must occur exactly 18 times"
        )
    source_counts = counts.loc[[key not in target_keys for key in counts.index]]
    if not source_counts.eq(1).all():
        raise RuntimeError("Source-prior triplets must remain unique")

    decades = np.floor(np.log10(train_values[:, 0])).astype(np.int64)
    decade_counts = pd.Series(decades).value_counts().sort_index()
    source_decade_counts = pd.Series(decades[source_mask]).value_counts().sort_index()
    if float(decade_counts.max() / len(train)) > 0.30:
        raise RuntimeError("A Young's-modulus decade exceeds 30% of training rows")
    if float(source_decade_counts.max() / source_mask.sum()) > 0.20:
        raise RuntimeError("A Young's-modulus decade exceeds 20% of source-prior rows")
    source_families = train.loc[source_mask, "material_family"].value_counts()
    if int(source_mask.sum()) != 44_645:
        raise RuntimeError("Source-prior row count changed after writing")
    if float(source_families.max() / 44_645) > 0.11:
        raise RuntimeError("A material family exceeds 11% of source-prior rows")
    if set(train_source["material_family"]) != set(original_source["material_family"]):
        raise RuntimeError("A source material family was removed from training")

    return {
        "training_rows": len(train),
        "normalized_unique_training_triplets": len(multiplicity),
        "canonical_target_triplets": len(target_keys),
        "target_repeats": 18,
        "source_rows": len(train_source),
        "complete_training_source_records": len(train_records),
        "held_out_source_rows": len(held_source),
        "validation_rows": len(validation),
        "validation_records": len(validation_records),
        "validation_families": int(validation["material_family"].nunique()),
        "validation_couplings": sorted(validation_couplings),
        "validation_max_rows_per_record": int(validation_sizes.max()),
        "source_record_overlap": len(train_records & validation_records),
        "effective_sample_size": effective_size,
        "max_multiplicity": int(multiplicity.max()),
        "youngs_decade_rows": {str(key): int(value) for key, value in decade_counts.items()},
        "source_youngs_decade_rows": {
            str(key): int(value) for key, value in source_decade_counts.items()
        },
        "max_source_family_fraction": float(
            source_families.max() / 44_645
        ),
        "normalization_bounds": dict(
            zip(("E_min", "E_max", "nu_min", "nu_max", "rho_min", "rho_max"), bounds)
        ),
    }


def main():
    args = parse_args()
    latent_dir = args.latent_dir.resolve()
    source_path = latent_dir / "material_ranges.csv"
    output_path = latent_dir / "materials_filtered_vae.csv"
    validation_path = latent_dir / "materials_validation_vae.csv"

    existing = pd.read_csv(source_path)
    protected = existing["protected_train"].fillna(False).astype(bool)
    targets = np.unique(triplet_array(existing.loc[protected]), axis=0)
    if len(targets) != 2_573:
        raise RuntimeError(
            f"Expected 2573 protected targets, found {len(targets)}"
        )
    source = existing.loc[~protected].copy()
    source_output, held_source, held_records, swaps = split_validation_records(source)
    source_output["split"] = "train"
    source_output["protected_train"] = False
    targets_output = target_frame(targets, source_output.columns)
    training = pd.concat([source_output, targets_output], ignore_index=True)
    training = training.sample(frac=1.0, random_state=42).reset_index(drop=True)
    training.to_csv(output_path, index=False, float_format="%.12g")

    validation = sample_validation(held_source)
    validation["split"] = "validation"
    validation["protected_train"] = False
    validation = validation.sample(frac=1.0, random_state=42).reset_index(drop=True)
    validation.to_csv(validation_path, index=False, float_format="%.12g")
    summary = validate_written(
        output_path,
        validation_path,
        targets,
        original_source=source,
        held_records=held_records,
    )
    summary["coupling_coverage_swaps"] = swaps
    print(pd.Series(summary).to_string(), flush=True)


if __name__ == "__main__":
    main()
