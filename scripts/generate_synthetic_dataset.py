from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from src.camera_utils import camera_records, create_turntable_cameras, linspace_angles, write_metadata
from src.config import ensure_pytorch3d, load_config, select_device, set_seed
from src.mesh_io import load_original_mesh, save_transform
from src.renderer import ReconstructionRenderer
from src.visualization import save_contact_sheet, save_image


def generate_synthetic_dataset() -> None:
    ensure_pytorch3d()
    cfg = load_config()
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    out_dir = cfg.dataset_synth_dir
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    mesh, transform = load_original_mesh(
        cfg.original_obj_path,
        device,
        center_normalize=cfg.mesh_center_normalize,
        scale_normalize=cfg.mesh_scale_normalize,
    )
    azimuths = linspace_angles(cfg.synth_azimuth_start_deg, cfg.synth_azimuth_end_deg, cfg.synth_num_views)
    cameras = create_turntable_cameras(azimuths, cfg.synth_elevation_deg, cfg.synth_camera_distance, device)
    renderer = ReconstructionRenderer(cfg.image_size, device, cfg.background_rgb)

    saved_images: list[Path] = []
    try:
        for i in tqdm(range(cfg.synth_num_views), desc="render synthetic"):
            cam = cameras[i]
            rgb = renderer.render_rgb(mesh, cam, soft=False)
            mask = renderer.render_mask(mesh, cam)
            image_path = images_dir / f"image_{i:03d}.png"
            mask_path = masks_dir / f"image_{i:03d}_mask.png"
            save_image(rgb, image_path)
            save_image((mask > 0.5).float(), mask_path)
            saved_images.append(image_path)
    except Exception as exc:
        if device.type != "mps":
            raise
        print(f"Warning: synthetic rendering failed on MPS ({exc}). Restarting on CPU.")
        device = torch.device("cpu")
        mesh, transform = load_original_mesh(
            cfg.original_obj_path,
            device,
            center_normalize=cfg.mesh_center_normalize,
            scale_normalize=cfg.mesh_scale_normalize,
        )
        cameras = create_turntable_cameras(azimuths, cfg.synth_elevation_deg, cfg.synth_camera_distance, device)
        renderer = ReconstructionRenderer(cfg.image_size, device, cfg.background_rgb)
        saved_images.clear()
        for i in tqdm(range(cfg.synth_num_views), desc="render synthetic cpu"):
            cam = cameras[i]
            rgb = renderer.render_rgb(mesh, cam, soft=False)
            mask = renderer.render_mask(mesh, cam)
            image_path = images_dir / f"image_{i:03d}.png"
            mask_path = masks_dir / f"image_{i:03d}_mask.png"
            save_image(rgb, image_path)
            save_image((mask > 0.5).float(), mask_path)
            saved_images.append(image_path)

    records = camera_records(cameras, azimuths, cfg.synth_elevation_deg, cfg.synth_camera_distance, cfg.image_size)
    write_metadata(
        out_dir / "metadata.json",
        records,
        {
            "source_obj": str(cfg.original_obj_path),
            "num_views": cfg.synth_num_views,
            "rotation_start_deg": cfg.synth_rotation_start_deg,
            "rotation_end_deg": cfg.synth_rotation_end_deg,
            "background_color": cfg.background_color,
        },
    )
    save_transform(transform, out_dir / "mesh_transform.json")
    save_contact_sheet(saved_images, out_dir / "preview.jpg")
    print(f"Synthetic dataset written to {out_dir}")


if __name__ == "__main__":
    generate_synthetic_dataset()
