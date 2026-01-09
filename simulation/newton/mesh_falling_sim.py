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
import numpy as np
import warp as wp

import newton


def main():
    parser = argparse.ArgumentParser(
        description="High-res soft cube fall with SemiImplicit solver"
    )
    parser.add_argument(
        "--grid_dim",
        type=int,
        default=48,
        help="Grid cells per axis (higher -> slower, more accurate)",
    )
    parser.add_argument(
        "--cell_size", type=float, default=0.1, help="Cell size (meters)"
    )
    parser.add_argument(
        "--drop_height", type=float, default=1.0, help="Drop height (meters)"
    )
    parser.add_argument("--fps", type=int, default=60, help="Viewer FPS")
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Seconds to simulate before auto-restart",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=256,
        help="Substeps per frame (smaller dt; higher -> slower)",
    )
    parser.add_argument(
        "--youngs_modulus",
        type=float,
        default=8000.0,
        help="Young's modulus (stiffness)",
    )
    parser.add_argument(
        "--poisson_ratio", type=float, default=0.45, help="Poisson's ratio (0-0.5)"
    )
    parser.add_argument(
        "--density", type=float, default=200.0, help="Material volumetric density"
    )
    parser.add_argument(
        "--materials",
        type=str,
        default=None,
        help="Path to NPZ with per-tet materials (E, nu, density)",
    )

    args = parser.parse_args()

    wp.init()

    fps = args.fps
    frame_dt = 1.0 / fps
    sim_substeps = max(1, args.substeps)
    sim_dt = frame_dt / sim_substeps
    total_frames = int(args.duration * fps)

    # Material (Lamé)
    E = args.youngs_modulus
    nu = args.poisson_ratio
    k_mu = 0.5 * E / (1.0 + nu)
    k_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    k_damp = 25.0

    print(f"Material: E={E:.1f}, nu={nu:.3f} -> mu={k_mu:.2f}, lambda={k_lambda:.2f}")

    builder = newton.ModelBuilder()

    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(ke=5.0e3, kd=50.0, mu=0.6)
    )

    dim = args.grid_dim
    cell = args.cell_size

    print(f"Grid: dim={dim}, cell={cell} m, cube extent≈{dim*cell:.3f} m")

    builder.default_particle_radius = cell * 0.5

    # this will be overridden by the materials file
    particle_density = float(args.density)

    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, float(args.drop_height)),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=cell,
        cell_y=cell,
        cell_z=cell,
        density=particle_density,
        k_mu=float(k_mu),
        k_lambda=float(k_lambda),
        k_damp=float(k_damp),
        tri_ke=1e-3,
        tri_ka=1e-3,
        tri_kd=1e-4,
        fix_bottom=False,
    )

    model = builder.finalize()
    print(f"Model: particles={model.particle_count} tets={model.tet_count}")

    # Soft contact parameters
    model.soft_contact_ke = 5.0e3
    model.soft_contact_kd = 50.0
    model.soft_contact_mu = 0.6

    # spatially varying tet materials and density
    # See: https://newton-physics.github.io/newton/api/_generated/newton.Model.html#newton.Model.tet_materials
    if args.materials:
        try:
            import numpy as _np
            import warp as _wp

            data = _np.load(args.materials)
            tet_count = int(model.tet_count)

            if "E" in data and "nu" in data:
                E_arr = _np.asarray(data["E"]).reshape(-1)
                nu_arr = _np.asarray(data["nu"]).reshape(-1)
                if E_arr.shape[0] == tet_count and nu_arr.shape[0] == tet_count:
                    mu_arr = 0.5 * E_arr / (1.0 + nu_arr)
                    lam_arr = (E_arr * nu_arr) / ((1.0 + nu_arr) * (1.0 - 2.0 * nu_arr))
                    damp_arr = _np.full(tet_count, k_damp, dtype=_np.float32)
                    tet_mats = _np.stack([mu_arr, lam_arr, damp_arr], axis=1).astype(
                        _np.float32
                    )
                    model.tet_materials = _wp.array(tet_mats, dtype=_wp.float32)
                else:
                    try:
                        est_dim = int(round(((E_arr.shape[0] / 5.0) ** (1.0 / 3.0))))
                    except Exception:
                        est_dim = -1
                    print(
                        f"[materials] E/nu length mismatch (got {E_arr.shape[0]}/{nu_arr.shape[0]}, need {tet_count}).\n"
                        f"  Hint: NPZ likely built for grid_dim≈{est_dim}; current grid_dim={dim}."
                    )

            rho_key = (
                "density" if "density" in data else ("rho" if "rho" in data else None)
            )
            if rho_key is not None:
                rho_arr = _np.asarray(data[rho_key]).reshape(-1)
                if rho_arr.shape[0] == tet_count:
                    pos = state_0.particle_q.numpy()
                    tet_idx = model.tet_indices.numpy().reshape(-1, 4)

                    def tet_vol(a, b, c, d):
                        return (
                            abs(
                                _np.linalg.det(_np.stack([b - a, c - a, d - a], axis=1))
                            )
                            / 6.0
                        )

                    vols = _np.empty(tet_count, dtype=_np.float64)
                    for t in range(tet_count):
                        i, j, k, l = tet_idx[t]
                        vols[t] = tet_vol(pos[i], pos[j], pos[k], pos[l])
                    p_mass = _np.zeros(model.particle_count, dtype=_np.float64)
                    for t in range(tet_count):
                        i, j, k, l = tet_idx[t]
                        m = float(rho_arr[t]) * float(vols[t]) / 4.0
                        p_mass[i] += m
                        p_mass[j] += m
                        p_mass[k] += m
                        p_mass[l] += m
                    model.particle_mass = _wp.array(
                        p_mass.astype(_np.float32), dtype=_wp.float32
                    )
                else:
                    try:
                        est_dim_rho = int(
                            round(((rho_arr.shape[0] / 5.0) ** (1.0 / 3.0)))
                        )
                    except Exception:
                        est_dim_rho = -1
                    print(
                        f"[materials] density length mismatch (got {rho_arr.shape[0]}, need {tet_count}).\n"
                        f"  Hint: NPZ likely built for grid_dim≈{est_dim_rho}; current grid_dim={dim}."
                    )
        except Exception as e:
            print(f"[materials] failed to apply materials: {e}")

    # SemiImplicit solver
    solver = newton.solvers.SolverSemiImplicit(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = model.collide(state_0, soft_contact_margin=0.01)

    viewer = newton.viewer.ViewerGL(headless=False)
    viewer.set_model(model)
    print("Viewer: SPACE to pause/resume, ESC to quit")

    sim_time = 0.0
    frame = 0

    while viewer.is_running():
        if frame >= total_frames:
            frame = 0
            sim_time = 0.0
            state_0 = model.state()
            state_1 = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        if not viewer.is_paused():
            for _ in range(sim_substeps):
                state_0.clear_forces()
                contacts = model.collide(state_0, soft_contact_margin=0.01)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

            sim_time += frame_dt
            frame += 1

            if frame % fps == 0:
                com = np.mean(state_0.particle_q.numpy(), axis=0)
                print(
                    f"t={sim_time:.2f}s  COM=({com[0]:.3f},{com[1]:.3f},{com[2]:.3f})"
                )

        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

    viewer.close()


if __name__ == "__main__":
    main()
