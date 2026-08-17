from __future__ import annotations

from dataclasses import dataclass

import torch
from pytorch3d.renderer import (
    AmbientLights,
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


@dataclass
class RenderOutput:
    rgb: torch.Tensor
    mask: torch.Tensor


def make_blend_params(background_rgb: tuple[float, float, float]) -> BlendParams:
    return BlendParams(sigma=1e-4, gamma=1e-4, background_color=background_rgb)


class ReconstructionRenderer:
    def __init__(
        self,
        image_size: int,
        device: torch.device,
        background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.image_size = image_size
        self.device = device
        self.background_rgb = background_rgb
        self.blend_params = make_blend_params(background_rgb)
        self.raster_settings_rgb = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.raster_settings_silhouette = RasterizationSettings(
            image_size=image_size,
            blur_radius=1e-4,
            faces_per_pixel=50,
        )
        self.lights = PointLights(device=device, location=[[0.0, 2.0, -3.0]])
        self.ambient_lights = AmbientLights(device=device)

    def _rgb_renderer(self, cameras: FoVPerspectiveCameras, soft: bool = True) -> MeshRenderer:
        shader_cls = SoftPhongShader if soft else HardPhongShader
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_rgb),
            shader=shader_cls(
                device=self.device,
                cameras=cameras,
                lights=self.lights,
                blend_params=self.blend_params,
            ),
        )

    def _mask_renderer(self, cameras: FoVPerspectiveCameras) -> MeshRenderer:
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_silhouette),
            shader=SoftSilhouetteShader(blend_params=self.blend_params),
        )

    def render_rgb(self, mesh: Meshes, cameras: FoVPerspectiveCameras, soft: bool = True) -> torch.Tensor:
        return self._call_with_mps_fallback(lambda: self._rgb_renderer(cameras, soft=soft)(mesh))[..., :3]

    def render_mask(self, mesh: Meshes, cameras: FoVPerspectiveCameras) -> torch.Tensor:
        rgba = self._call_with_mps_fallback(lambda: self._mask_renderer(cameras)(mesh))
        return rgba[..., 3:4]

    def render(self, mesh: Meshes, cameras: FoVPerspectiveCameras) -> RenderOutput:
        rgb = self.render_rgb(mesh, cameras)
        mask = self.render_mask(mesh, cameras)
        return RenderOutput(rgb=rgb, mask=mask)

    def _call_with_mps_fallback(self, fn):
        try:
            return fn()
        except Exception as exc:
            if self.device.type != "mps":
                raise
            print(f"Warning: PyTorch3D rendering failed on MPS ({exc}). Retrying on CPU.")
            self.device = torch.device("cpu")
            self.lights = PointLights(device=self.device, location=[[0.0, 2.0, -3.0]])
            return fn()
