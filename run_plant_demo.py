import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

from vomp.inference import Vomp
from vomp.inference.utils import save_materials
from vomp.representations.gaussian.gaussian_model import Gaussian

PLY = "filtered_gaussians_plant.ply"
OUT = "outputs/plant_demo"

model = Vomp.from_checkpoint(config_path="weights/inference.json", use_trt=False)
g = Gaussian(sh_degree=3, aabb=[-1, -1, -1, 2, 2, 2], device="cuda")
g.load_ply(PLY)
results = model.get_splat_materials(g, output_dir=OUT, seed=42)
save_materials(results, f"{OUT}/materials.npz")

xyz = g.get_xyz.detach().cpu().numpy()
E = np.asarray(results["youngs_modulus"], float)
nu = np.asarray(results["poisson_ratio"], float)
rho = np.asarray(results["density"], float)

# --- visualization ---
props = [
    ("Young's modulus  E [Pa] (log)", np.log10(E), "viridis", True),
    ("Poisson ratio  nu", nu, "plasma", False),
    ("Density  rho [kg/m^3]", rho, "cividis", False),
]
fig = plt.figure(figsize=(13.5, 5.0))
for c, (title, cval, cmap, log) in enumerate(props):
    vmin, vmax = cval.min(), np.percentile(cval, 95)
    norm = mcolors.Normalize(vmin, vmax)
    ax = fig.add_subplot(1, 3, c + 1, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=cval, cmap=cmap, norm=norm,
               s=1.6, marker=".", lw=0, depthshade=False)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=12, azim=-60)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, weight="bold")
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.03, pad=0.0, orientation="horizontal")
    if log:
        cb.set_ticks(np.linspace(vmin, vmax, 4))
        cb.set_ticklabels([f"{10**t/1e9:.1f}G" for t in np.linspace(vmin, vmax, 4)])
fig.savefig(f"{OUT}/materials.png", dpi=140, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}/materials.png  and  {OUT}/materials.npz")
