import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

from vomp.inference import Vomp
from vomp.representations.gaussian.gaussian_model import (
    Gaussian,
    transform_xyz,
    transform_rot,
    transform_shs,
)

PLY = "filtered_gaussians_plant.ply"
model = Vomp.from_checkpoint(config_path="weights/inference.json", use_trt=False)


def load():
    g = Gaussian(sh_degree=3, aabb=[-1, -1, -1, 2, 2, 2], device="cuda")
    g.load_ply(PLY)
    return g


def rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def reorient(g, R):
    Rt = torch.tensor(R, dtype=torch.float32, device=g.get_xyz.device)
    T = torch.eye(4, device=Rt.device)
    T[:3, :3] = Rt
    g.from_xyz(transform_xyz(g.get_xyz.detach().clone(), T))
    g.from_rotation(transform_rot(g.get_rotation.detach().clone(), T))
    if g._features_rest is not None and g._features_rest.shape[1] > 0:
        g._features_rest = transform_shs(g._features_rest.detach().clone(), Rt)
    return g


X, Y, Z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
ORIENTS = {
    "original orientation": np.eye(3),
    "z180": rot(Z, 180),
    "z90": rot(Z, 90),
    "z120": rot(Z, 120),
    "z60": rot(Z, 60),
    "z-15": rot(Z, -15),
    "z30y10": rot(Y, 10) @ rot(Z, 30),
    "z20x10": rot(X, 10) @ rot(Z, 20),
    "y15": rot(Y, 15),
    "x15": rot(X, 15),
    "x20": rot(X, 20),
}

# --- run inference at each orientation ---
# i have changed output directory so it is not using cached results
mats = {}
for name, R in ORIENTS.items():
    g = load()
    if not np.allclose(R, np.eye(3)):
        reorient(g, R)
    model._view_keep = name  # per-orientation view ids to keep (view_dirs.json)
    r = model.get_splat_materials(g, output_dir=f"outputs/orient_{name}", seed=42)
    mats[name] = np.column_stack(
        [r["youngs_modulus"], r["poisson_ratio"], r["density"]]
    )

# --- for figure show all of them in same orientation ---
g0 = load()
xyz = g0.get_xyz.detach().cpu().numpy()
R_align = np.array([[1.00, -0.01, 0.00], [-0.01, -0.82, -0.57], [0.00, 0.57, -0.82]])
P = (xyz - xyz.mean(0)) @ R_align.T
lim = 0.02

# --- figure ---
rows = [
    ("Young's E", 0, True, "viridis"),
    ("Poisson nu", 1, False, "plasma"),
    ("Density rho", 2, False, "cividis"),
]
names = list(ORIENTS)
nc = len(names)
fig = plt.figure(figsize=(2.5 * nc + 0.5, 3.1 * 3))
gs = fig.add_gridspec(
    3, nc + 1, width_ratios=[1] * nc + [0.04], wspace=0.02, hspace=0.05
)
for ri, (rlabel, idx, log, cmap) in enumerate(rows):
    allv = np.concatenate(
        [(np.log10(mats[n][:, idx]) if log else mats[n][:, idx]) for n in names]
    )
    vmin, vmax = np.percentile(allv, [2, 97 if log else 98])
    norm = mcolors.Normalize(vmin, vmax)
    for ci, n in enumerate(names):
        val = np.log10(mats[n][:, idx]) if log else mats[n][:, idx]
        ax = fig.add_subplot(gs[ri, ci], projection="3d")
        ax.scatter(
            P[:, 0],
            P[:, 1],
            P[:, 2],
            c=val,
            cmap=cmap,
            norm=norm,
            s=1.0,
            marker=".",
            lw=0,
            depthshade=False,
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=12, azim=-60)
        ax.set_axis_off()
        if ri == 0:
            ax.set_title(n, fontsize=10, weight="bold")
        if ci == 0:
            ax.text2D(
                -0.06,
                0.5,
                rlabel,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=11,
                weight="bold",
            )
    cax = fig.add_subplot(gs[ri, nc])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    if log:
        cb.set_ticks(np.linspace(vmin, vmax, 4))
        cb.set_ticklabels([f"{10**t/1e9:.1f}G" for t in np.linspace(vmin, vmax, 4)])
fig.savefig(
    "outputs/plant_orientations_demo.png",
    dpi=120,
    bbox_inches="tight",
    facecolor="white",
)
print("saved outputs/plant_orientations_demo.png")
