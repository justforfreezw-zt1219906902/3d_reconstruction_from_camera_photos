from __future__ import annotations

import torch
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes


class MPSRendererUnsupported(RuntimeError):
    pass


class ReconstructionRenderer:
    def __init__(self, image_size: int, device: torch.device, faces_per_pixel: int = 20) -> None:
        self.image_size = image_size
        self.device = device
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1.0, 1.0, 1.0))
        self.rgb_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        self.silhouette_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=1e-4,
            faces_per_pixel=faces_per_pixel,
        )
        self.lights = PointLights(device=device, location=[[0.0, 2.0, -3.0]])
        self._silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.silhouette_settings),
            shader=SoftSilhouetteShader(blend_params=self.blend_params),
        )
        self._rgb_renderers = {
            True: MeshRenderer(
                rasterizer=MeshRasterizer(raster_settings=self.rgb_settings),
                shader=SoftPhongShader(
                    device=device,
                    lights=self.lights,
                    blend_params=self.blend_params,
                ),
            ),
            False: MeshRenderer(
                rasterizer=MeshRasterizer(raster_settings=self.rgb_settings),
                shader=HardPhongShader(
                    device=device,
                    lights=self.lights,
                    blend_params=self.blend_params,
                ),
            ),
        }

    def _raise_mps(self, exc: Exception) -> None:
        if self.device.type == "mps":
            raise MPSRendererUnsupported(
                "PyTorch3D rasterization is not supported for this MPS operation; restart the full pipeline on CPU."
            ) from exc
        raise exc

    def render_mask(self, mesh: Meshes, cameras: FoVPerspectiveCameras) -> torch.Tensor:
        render_mesh = self._mesh_for_cameras(mesh, cameras)
        try:
            return self._silhouette_renderer(render_mesh, cameras=cameras)[..., 3:4]
        except Exception as exc:
            self._raise_mps(exc)
            raise

    def render_rgb(self, mesh: Meshes, cameras: FoVPerspectiveCameras, soft: bool = True) -> torch.Tensor:
        render_mesh = self._mesh_for_cameras(mesh, cameras)
        try:
            return self._rgb_renderers[soft](render_mesh, cameras=cameras)[..., :3]
        except Exception as exc:
            self._raise_mps(exc)
            raise

    @staticmethod
    def _mesh_for_cameras(mesh: Meshes, cameras: FoVPerspectiveCameras) -> Meshes:
        camera_count = int(cameras.R.shape[0])
        mesh_count = len(mesh)
        if mesh_count == camera_count:
            return mesh
        if mesh_count == 1:
            return mesh.extend(camera_count)
        raise ValueError(f"Mesh batch ({mesh_count}) does not match camera batch ({camera_count}).")

    def probe(self, mesh: Meshes, cameras: FoVPerspectiveCameras) -> None:
        """Run one rasterization before optimization so unsupported MPS fails early."""
        self.render_mask(mesh, cameras)
