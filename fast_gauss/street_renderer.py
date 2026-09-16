"""StreetWorld draw orchestration on top of the layered batch renderer."""

from typing import TYPE_CHECKING, Sequence

import os
import time

import torch

from .cuda_utils import CHECK_CUDART_ERROR

import OpenGL.GL as gl

if TYPE_CHECKING:
    from . import GaussianRasterizationSettings


class StreetRendererMixin:
    @torch.no_grad()
    def render_street_batch(
        self,
        means3d: torch.Tensor,
        cov6: torch.Tensor,
        rgb3: torch.Tensor,
        occ1: torch.Tensor,
        visibility: torch.Tensor,
        depths: torch.Tensor,
        raster_settings: Sequence['GaussianRasterizationSettings'],
    ):
        from . import street_pipeline

        profile = os.environ.get('FAST_GAUSS_BATCH_PROFILE') == '1'
        timings = {}
        if profile:
            torch.cuda.synchronize()
            started_at = time.perf_counter()

        draw_gids = []
        draw_counts = []
        for camera_id in range(len(raster_settings)):
            gids = visibility[camera_id].nonzero()[..., 0]
            order = depths[camera_id][gids].argsort(descending=True, stable=True)
            draw_gids.append(gids[order].to(torch.int32))
            draw_counts.append(gids.numel())
        draw_gids = torch.cat(draw_gids)
        draw_counts = torch.tensor(draw_counts, dtype=torch.int32, device=visibility.device)
        if profile:
            torch.cuda.synchronize()
            timings['street_compact_and_sort'] = time.perf_counter() - started_at
            started_at = time.perf_counter()

        self.resize_batch_draw_commands(len(raster_settings))
        height, width = self._prepare_batch(draw_gids.numel(), raster_settings)
        if profile:
            timings['gl_setup_and_uniforms'] = time.perf_counter() - started_at
            started_at = time.perf_counter()

        from cuda import cudart

        stream = torch.cuda.current_stream().cuda_stream
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.batch_cu_vbo, stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.batch_cu_draw_commands, stream))
        vbo_ptr, _ = CHECK_CUDART_ERROR(cudart.cudaGraphicsResourceGetMappedPointer(self.batch_cu_vbo))
        draw_command_ptr, _ = CHECK_CUDART_ERROR(
            cudart.cudaGraphicsResourceGetMappedPointer(self.batch_cu_draw_commands)
        )
        if profile:
            torch.cuda.synchronize()
            timings['cuda_gl_vbo_map'] = time.perf_counter() - started_at
            started_at = time.perf_counter()
        street_pipeline.pack_vbo(vbo_ptr, means3d, cov6, rgb3, occ1, draw_gids, draw_counts)
        street_pipeline.write_draw_commands(draw_command_ptr, draw_counts)
        if profile:
            torch.cuda.synchronize()
            timings['cuda_vbo_pack'] = time.perf_counter() - started_at
            started_at = time.perf_counter()
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.batch_cu_vbo, stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.batch_cu_draw_commands, stream))
        if profile:
            torch.cuda.synchronize()
            timings['cuda_gl_vbo_unmap'] = time.perf_counter() - started_at
            started_at = time.perf_counter()

        self._begin_batch_draw(height, width)
        gl.glBindBuffer(gl.GL_DRAW_INDIRECT_BUFFER, self.batch_draw_commands)
        gl.glMultiDrawArraysIndirect(gl.GL_POINTS, None, len(raster_settings), 0)
        gl.glBindVertexArray(0)
        if profile:
            gl.glFinish()
            timings['layered_draw'] = time.perf_counter() - started_at
            started_at = time.perf_counter()

        rgba = self._read_batch_texture(len(raster_settings), height, width)
        if profile:
            torch.cuda.synchronize()
            timings['cuda_gl_readback'] = time.perf_counter() - started_at
            self.last_batch_profile = timings

        return self._cast_batch_output(rgba, means3d.dtype)


    def rasterize_street_gaussians_batch(
        self,
        raw,
        raster_settings: Sequence['GaussianRasterizationSettings'],
    ):
        from . import street_pipeline

        camera_centers = torch.stack([
            torch.as_tensor(setting.campos, dtype=raw["xyz"].dtype, device=raw["xyz"].device)
            for setting in raster_settings
        ])
        projections = torch.stack([
            torch.as_tensor(setting.projmatrix, dtype=raw["xyz"].dtype, device=raw["xyz"].device)
            for setting in raster_settings
        ])
        view_matrices = torch.stack([
            torch.as_tensor(setting.viewmatrix, dtype=raw["xyz"].dtype, device=raw["xyz"].device)
            for setting in raster_settings
        ])
        means3d, covariances, opacities, colors, depths, visibility = street_pipeline.prepare(
            raw | {"projections": projections, "view_matrices": view_matrices},
            camera_centers,
        )
        rgba = self.render_street_batch(
            means3d,
            covariances,
            colors,
            opacities,
            visibility,
            depths,
            raster_settings,
        )
        image, alpha = rgba[..., :3].permute(0, 3, 1, 2), rgba[..., 3:].permute(0, 3, 1, 2)
        if os.environ.get('FAST_GAUSS_BATCH_PROFILE') == '1':
            self.last_batch_profile = {**street_pipeline.last_profile, **self.last_batch_profile}
        return image.float(), alpha.float()
