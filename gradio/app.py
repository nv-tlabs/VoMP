import glob
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import numpy as np
import spaces
import torch

from vomp.inference import Vomp
from vomp.inference.utils import save_materials

NUM_VIEWS = 150
PROPERTY_NAMES = ["youngs_modulus", "poissons_ratio", "density"]
PROPERTY_DISPLAY_NAMES = {
    "youngs_modulus": "Young's Modulus",
    "poissons_ratio": "Poisson's Ratio",
    "density": "Density",
}

BLENDER_LINK = (
    "https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz"
)
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender"

EXAMPLES_DIR = "examples"


def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        print("Installing Blender...")
        os.system("sudo apt-get update")
        os.system(
            "sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6"
        )
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(
            f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}"
        )
        print("Blender installed successfully!")


def _is_gaussian_splat(file_path: str) -> bool:
    if not file_path.lower().endswith(".ply"):
        return False

    try:
        with open(file_path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
                if len(header) > 10000:
                    break

            header_str = header.decode("utf-8", errors="ignore").lower()
            gaussian_indicators = ["f_dc", "opacity", "scale_0", "rot_0"]
            return any(indicator in header_str for indicator in gaussian_indicators)
    except Exception:
        return False


def _setup_examples():
    """Ensure examples directory exists."""
    os.makedirs(EXAMPLES_DIR, exist_ok=True)


_setup_examples()


print("Loading VoMP model...")
model = Vomp.from_checkpoint(
    config_path="weights/inference.json",
    geometry_checkpoint_dir="weights/geometry_transformer.pt",
    matvae_checkpoint_dir="weights/matvae.safetensors",
    normalization_params_path="weights/normalization_params.json",
)
print("VoMP model loaded successfully!")


def _get_render_images(output_dir: str) -> List[str]:
    renders_dir = os.path.join(output_dir, "renders")
    if not os.path.exists(renders_dir):
        return []
    image_paths = sorted(glob.glob(os.path.join(renders_dir, "*.png")))
    return image_paths


def _create_colorbar(
    data: np.ndarray, property_name: str, output_path: str, colormap: str = "viridis"
) -> str:
    fig, ax = plt.subplots(figsize=(6, 0.8))
    fig.subplots_adjust(bottom=0.5)
    ax.remove()

    cmap = plt.cm.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

    cbar_ax = fig.add_axes([0.1, 0.4, 0.8, 0.35])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.ax.set_xlabel(
        f"{PROPERTY_DISPLAY_NAMES.get(property_name, property_name)}", fontsize=10
    )

    plt.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="white", transparent=False
    )
    plt.close()
    return output_path


def _render_point_cloud_views(
    coords: np.ndarray,
    values: np.ndarray,
    output_dir: str,
    property_name: str,
    colormap: str = "viridis",
) -> List[str]:
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin > 1e-10:
        normalized = (values - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(values)

    cmap = plt.cm.get_cmap(colormap)
    colors = cmap(normalized)

    views = [
        (30, 45, "view1"),
        (30, 135, "view2"),
        (80, 45, "view3"),
    ]

    image_paths = []

    for elev, azim, view_name in views:
        fig = plt.figure(figsize=(6, 6), facecolor="#1a1a1a")
        ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=colors,
            s=15,
            alpha=0.9,
        )

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([-0.6, 0.6])
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

        output_path = os.path.join(output_dir, f"{property_name}_{view_name}.png")
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="#1a1a1a",
            edgecolor="none",
        )
        plt.close()

        image_paths.append(output_path)

    return image_paths


def _create_material_visualizations(
    material_file: str, output_dir: str
) -> Dict[str, Tuple[List[str], str]]:
    result = {}
    data = np.load(material_file, allow_pickle=True)

    if "voxel_data" in data:
        voxel_data = data["voxel_data"]
        coords = np.column_stack([voxel_data["x"], voxel_data["y"], voxel_data["z"]])
        properties = {
            "youngs_modulus": voxel_data["youngs_modulus"],
            "poissons_ratio": voxel_data["poissons_ratio"],
            "density": voxel_data["density"],
        }
    else:
        if "voxel_coords_world" in data:
            coords = data["voxel_coords_world"]
        elif "query_coords_world" in data:
            coords = data["query_coords_world"]
        elif "coords" in data:
            coords = data["coords"]
        else:
            print(f"Warning: No coordinate data found in {material_file}")
            return result

        properties = {}
        property_mapping = {
            "youngs_modulus": ["youngs_modulus", "young_modulus"],
            "poissons_ratio": ["poissons_ratio", "poisson_ratio"],
            "density": ["density"],
        }
        for prop_name, possible_names in property_mapping.items():
            for name in possible_names:
                if name in data:
                    properties[prop_name] = data[name]
                    break

    center = (np.min(coords, axis=0) + np.max(coords, axis=0)) / 2
    max_range = np.max(np.max(coords, axis=0) - np.min(coords, axis=0))
    if max_range > 1e-10:
        coords_normalized = (coords - center) / max_range
    else:
        coords_normalized = coords - center

    for prop_name, prop_data in properties.items():
        if prop_data is not None:
            view_paths = _render_point_cloud_views(
                coords_normalized, prop_data, output_dir, prop_name
            )
            colorbar_path = os.path.join(output_dir, f"{prop_name}_colorbar.png")
            _create_colorbar(prop_data, prop_name, colorbar_path)
            result[prop_name] = (view_paths, colorbar_path)
            print(f"Created visualization for {prop_name}: {len(view_paths)} views")

    return result


@spaces.GPU(duration=60)
@torch.no_grad()
def process_3d_model(input_file):
    empty_result = (
        None,
        [],
        None,
        [],
        None,
        None,
        [],
        None,
        None,
        [],
        None,
        None,
    )

    if input_file is None:
        return empty_result

    output_dir = tempfile.mkdtemp(prefix="vomp_")
    material_file = os.path.join(output_dir, "materials.npz")

    try:
        if _is_gaussian_splat(input_file):
            print(f"Processing as Gaussian splat: {input_file}")
            results = model.get_splat_materials(
                input_file,
                voxel_method="kaolin",
                query_points="voxel_centers",
                output_dir=output_dir,
            )
        else:
            print(f"Processing as mesh: {input_file}")
            _install_blender()
            results = model.get_mesh_materials(
                input_file,
                blender_path=BLENDER_PATH,
                query_points="voxel_centers",
                output_dir=output_dir,
                return_original_scale=True,
            )

        save_materials(results, material_file)
        print(f"Materials saved to: {material_file}")

        all_images = _get_render_images(output_dir)
        first_image = all_images[0] if all_images else None

        visualizations = _create_material_visualizations(material_file, output_dir)

        youngs_views = visualizations.get("youngs_modulus", ([], None))[0]
        youngs_colorbar = visualizations.get("youngs_modulus", ([], None))[1]
        youngs_first = youngs_views[0] if youngs_views else None

        poissons_views = visualizations.get("poissons_ratio", ([], None))[0]
        poissons_colorbar = visualizations.get("poissons_ratio", ([], None))[1]
        poissons_first = poissons_views[0] if poissons_views else None

        density_views = visualizations.get("density", ([], None))[0]
        density_colorbar = visualizations.get("density", ([], None))[1]
        density_first = density_views[0] if density_views else None

        return (
            first_image,
            all_images,
            youngs_first,
            youngs_views,
            youngs_colorbar,
            poissons_first,
            poissons_views,
            poissons_colorbar,
            density_first,
            density_views,
            density_colorbar,
            material_file,
        )

    except Exception as e:
        print(f"Error processing 3D model: {e}")
        raise gr.Error(f"Failed to process 3D model: {str(e)}")


def update_slider_image(slider_value: int, all_images: List[str]) -> Optional[str]:
    if not all_images or slider_value < 0 or slider_value >= len(all_images):
        return None
    return all_images[slider_value]


def update_property_view(slider_value: int, views: List[str]) -> Optional[str]:
    if not views or slider_value < 0 or slider_value >= len(views):
        return None
    return views[slider_value]


css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
}

.title-container {
    text-align: center;
    padding: 20px 0;
}

.badge-container {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}

.badge-container a img {
    height: 22px;
}

h1 {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}

.input-column, .output-column {
    min-height: 400px;
}

.output-column .row {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 16px;
}

.output-column .row > .column {
    flex: 1 1 50% !important;
    min-width: 0 !important;
}
"""

title_md = """
<div class="title-container">
    <h1>VoMP: Predicting Volumetric Mechanical Properties</h1>
    <p class="subtitle">Feed-forward, fine-grained, physically based volumetric material properties from Splats, Meshes, NeRFs, and more.</p>
    <div class="badge-container">
        <a href="https://arxiv.org/abs/2510.22975"><img src="https://img.shields.io/badge/arXiv-VoMP-red" alt="Paper PDF"></a>
        <a href="https://research.nvidia.com/labs/sil/projects/vomp/"><img src="https://img.shields.io/badge/Project_Page-VoMP-green" alt="Project Page"></a>
        <a href="https://huggingface.co/nvidia/VoMP"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20-Models-yellow" alt="Models"></a>
        <a href="https://huggingface.co/datasets/nvidia/VoMP-GVM-Dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20-GVM%20Dataset-yellow" alt="GVM Dataset"></a>
        <a href="https://huggingface.co/datasets/nvidia/VoMP-MTD-Dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20-MTD%20Dataset-yellow" alt="MTD Dataset"></a>
    </div>
</div>
"""

description_md = """
Upload a Gaussian Splat (.ply) or Mesh (.obj, .glb, .stl, .gltf) to predict volumetric mechanical properties (Young's modulus, Poisson's ratio, density) for realistic physics simulation.
"""

with gr.Blocks(css=css, title="VoMP") as demo:
    all_images_state = gr.State([])
    youngs_views_state = gr.State([])
    poissons_views_state = gr.State([])
    density_views_state = gr.State([])

    gr.HTML(title_md)
    gr.Markdown(description_md)

    with gr.Row():
        # Input Column (50%)
        with gr.Column(scale=1, elem_classes="input-column"):
            gr.Markdown("### 📤 Input")
            input_model = gr.Model3D(
                label="Upload 3D Model",
                clear_color=[0.1, 0.1, 0.1, 1.0],
            )

            submit_btn = gr.Button(
                "🚀 Generate Materials", variant="primary", size="lg"
            )

            gr.Markdown("#### 🎬 Rendered Views")
            rendered_image = gr.Image(label="Rendered View", height=250)

            view_slider = gr.Slider(
                minimum=0,
                maximum=NUM_VIEWS - 1,
                step=1,
                value=0,
                label="Browse All Views",
                info=f"Slide to view all {NUM_VIEWS} rendered views",
            )

        # Output Column (50%)
        with gr.Column(scale=1, elem_classes="output-column"):
            gr.Markdown("### 📥 Output - Material Properties")

            # Row 1: Young's Modulus and Poisson's Ratio
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    youngs_image = gr.Image(label="Young's Modulus", height=200)
                    youngs_slider = gr.Slider(
                        minimum=0,
                        maximum=2,
                        step=1,
                        value=0,
                        label="View",
                        info="Switch between 3 views",
                    )
                    youngs_colorbar = gr.Image(height=50, show_label=False)

                with gr.Column(scale=1, min_width=200):
                    poissons_image = gr.Image(label="Poisson's Ratio", height=200)
                    poissons_slider = gr.Slider(
                        minimum=0,
                        maximum=2,
                        step=1,
                        value=0,
                        label="View",
                        info="Switch between 3 views",
                    )
                    poissons_colorbar = gr.Image(height=50, show_label=False)

            # Row 2: Density and Download
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    density_image = gr.Image(label="Density", height=200)
                    density_slider = gr.Slider(
                        minimum=0,
                        maximum=2,
                        step=1,
                        value=0,
                        label="View",
                        info="Switch between 3 views",
                    )
                    density_colorbar = gr.Image(height=50, show_label=False)

                with gr.Column(scale=1, min_width=200):
                    gr.Markdown("#### 💾 Download")
                    output_file = gr.File(
                        label="Download Materials (.npz)",
                        file_count="single",
                    )

    gr.Markdown("### 🎯 Examples")
    gr.Examples(
        examples=[
            [os.path.join(EXAMPLES_DIR, "dog.ply")],
            [os.path.join(EXAMPLES_DIR, "dozer.ply")],
            [os.path.join(EXAMPLES_DIR, "fiscus.ply")],
            [os.path.join(EXAMPLES_DIR, "plant.ply")],
        ],
        inputs=[input_model],
        outputs=[
            rendered_image,
            all_images_state,
            youngs_image,
            youngs_views_state,
            youngs_colorbar,
            poissons_image,
            poissons_views_state,
            poissons_colorbar,
            density_image,
            density_views_state,
            density_colorbar,
            output_file,
        ],
        fn=process_3d_model,
        cache_examples=False,
    )

    # Event handlers
    submit_btn.click(
        fn=process_3d_model,
        inputs=[input_model],
        outputs=[
            rendered_image,
            all_images_state,
            youngs_image,
            youngs_views_state,
            youngs_colorbar,
            poissons_image,
            poissons_views_state,
            poissons_colorbar,
            density_image,
            density_views_state,
            density_colorbar,
            output_file,
        ],
    )

    view_slider.change(
        fn=update_slider_image,
        inputs=[view_slider, all_images_state],
        outputs=[rendered_image],
    )

    youngs_slider.change(
        fn=update_property_view,
        inputs=[youngs_slider, youngs_views_state],
        outputs=[youngs_image],
    )

    poissons_slider.change(
        fn=update_property_view,
        inputs=[poissons_slider, poissons_views_state],
        outputs=[poissons_image],
    )

    density_slider.change(
        fn=update_property_view,
        inputs=[density_slider, density_views_state],
        outputs=[density_image],
    )

if __name__ == "__main__":
    demo.launch()
