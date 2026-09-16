"""Layered OpenGL renderer shared by generic and StreetWorld batches."""

from typing import TYPE_CHECKING, Sequence

import ctypes
import os
import time

import glm
import numpy as np
import torch
from glm import mat4

from .base_utils import dotdict
from .cuda_utils import CHECK_CUDART_ERROR
from .gaussian_utils import build_cov6
from .gl_utils import load_shader_source, use_gl_program
from .math_utils import normalize
from .sh_utils import eval_sh

import OpenGL.GL as gl
from OpenGL.GL import shaders

if TYPE_CHECKING:
    from . import GaussianRasterizationSettings


_BATCH_VBO_PACK_SOURCE = b'''extern "C" __global__ void pack_batch_vbo(
    float* output, const float* xyz, const float* cov, const float* rgb,
    const float* occ, const long long* order, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    long long source_index = order[index];
    float* target = output + index * 13;
    const float* source_xyz = xyz + source_index * 3;
    const float* source_cov = cov + source_index * 6;
    const float* source_rgb = rgb + source_index * 3;
    target[0] = source_xyz[0]; target[1] = source_xyz[1]; target[2] = source_xyz[2];
    target[3] = source_cov[0]; target[4] = source_cov[1]; target[5] = source_cov[2];
    target[6] = source_cov[3]; target[7] = source_cov[4]; target[8] = source_cov[5];
    target[9] = source_rgb[0]; target[10] = source_rgb[1]; target[11] = source_rgb[2];
    target[12] = occ[source_index];
}'''
_batch_vbo_pack_module = None
_batch_vbo_pack_kernel = None


def _check_cuda_driver(result, cuda):
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'CUDA driver API failed: {result[0]}')
    return result[1:]


def _check_nvrtc(result, nvrtc):
    if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f'NVRTC failed: {result[0]}')
    return result[1:]


def _get_batch_vbo_pack_kernel():
    global _batch_vbo_pack_module, _batch_vbo_pack_kernel
    if _batch_vbo_pack_kernel is not None:
        return _batch_vbo_pack_kernel

    from cuda import cuda, nvrtc

    major, minor = torch.cuda.get_device_capability()
    program, = _check_nvrtc(
        nvrtc.nvrtcCreateProgram(_BATCH_VBO_PACK_SOURCE, b'pack_batch_vbo.cu', 0, [], []), nvrtc
    )
    result, = nvrtc.nvrtcCompileProgram(
        program, 1, [f'--gpu-architecture=compute_{major}{minor}'.encode()]
    )
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log_size, = _check_nvrtc(nvrtc.nvrtcGetProgramLogSize(program), nvrtc)
        log_buffer = bytearray(log_size)
        _check_nvrtc(nvrtc.nvrtcGetProgramLog(program, log_buffer), nvrtc)
        raise RuntimeError(log_buffer.decode())
    ptx_size, = _check_nvrtc(nvrtc.nvrtcGetPTXSize(program), nvrtc)
    ptx = bytearray(ptx_size)
    _check_nvrtc(nvrtc.nvrtcGetPTX(program, ptx), nvrtc)
    _batch_vbo_pack_module, = _check_cuda_driver(cuda.cuModuleLoadData(bytes(ptx)), cuda)
    _batch_vbo_pack_kernel, = _check_cuda_driver(
        cuda.cuModuleGetFunction(_batch_vbo_pack_module, b'pack_batch_vbo'), cuda
    )
    return _batch_vbo_pack_kernel


def _pack_batch_vbo(vbo_pointer, xyz, cov, rgb, occ, order, stream):
    from cuda import cuda

    count = len(order)
    _check_cuda_driver(
        cuda.cuLaunchKernel(
            _get_batch_vbo_pack_kernel(),
            (count + 255) // 256,
            1,
            1,
            256,
            1,
            1,
            0,
            stream,
            (
                (
                    int(vbo_pointer), xyz.data_ptr(), cov.data_ptr(), rgb.data_ptr(), occ.data_ptr(),
                    order.data_ptr(), count,
                ),
                (
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                ),
            ),
            0,
        ),
        cuda,
    )




class BatchedRendererMixin:
    def compile_batch_shader(self):
        self.gsplat_batch_program = shaders.compileProgram(
            shaders.compileShader(load_shader_source('dsplat_batch_drawid.vert'), gl.GL_VERTEX_SHADER),
            shaders.compileShader(load_shader_source('dsplat_batch_drawid.geom'), gl.GL_GEOMETRY_SHADER),
            shaders.compileShader(load_shader_source('dsplat.frag'), gl.GL_FRAGMENT_SHADER),
        )

    def use_gl_batch_program(self):
        use_gl_program(self.gsplat_batch_program)
        self.batch_uniforms = dotdict(
            cameraCount=gl.glGetUniformLocation(self.gsplat_batch_program, "cameraCount"),
            P=gl.glGetUniformLocation(self.gsplat_batch_program, "uP[0]"),
            VM=gl.glGetUniformLocation(self.gsplat_batch_program, "uVM[0]"),
            focal=gl.glGetUniformLocation(self.gsplat_batch_program, "uFocal[0]"),
            principal=gl.glGetUniformLocation(self.gsplat_batch_program, "uPrincipal[0]"),
            basisViewport=gl.glGetUniformLocation(self.gsplat_batch_program, "uBasisViewport[0]"),
            useDepth=gl.glGetUniformLocation(self.gsplat_batch_program, "useDepth"),
            solidMode=gl.glGetUniformLocation(self.gsplat_batch_program, "solidMode"),
            edgeMode=gl.glGetUniformLocation(self.gsplat_batch_program, "edgeMode"),
        )


    def init_batch_gl_buffers(self, vertices: int):
        from cuda import cudart

        if hasattr(self, 'batch_cu_vbo'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.batch_cu_vbo))
        if hasattr(self, 'batch_vao'):
            gl.glDeleteVertexArrays(1, [self.batch_vao])
            gl.glDeleteBuffers(1, [self.batch_vbo])

        element_size = 4 if self.dtype == torch.float else 2
        attr_sizes = [3, 3, 3, 4]
        stride = sum(attr_sizes) * element_size
        self.batch_vao = gl.glGenVertexArrays(1)
        self.batch_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.batch_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.batch_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices * stride, ctypes.c_void_p(0), gl.GL_DYNAMIC_DRAW)
        offset = 0
        for index, size in enumerate(attr_sizes):
            gl.glVertexAttribPointer(index, size, self.gl_attr_dtypes[0], gl.GL_FALSE, stride, ctypes.c_void_p(offset * element_size))
            gl.glEnableVertexAttribArray(index)
            offset += size
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        self.batch_cu_vbo = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterBuffer(self.batch_vbo, flags))
        self.batch_max_vertices = vertices

    def init_batch_textures(self, height: int, width: int, camera_count: int):
        from cuda import cudart

        if hasattr(self, 'batch_cu_tex'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.batch_cu_tex))
        if hasattr(self, 'batch_fbo'):
            gl.glDeleteFramebuffers(1, [self.batch_fbo])
            gl.glDeleteTextures(2, [self.batch_rgba_tex, self.batch_depth_tex])

        self.batch_rgba_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.batch_rgba_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexImage3D(
            gl.GL_TEXTURE_2D_ARRAY,
            0,
            self.gl_tex_dtype,
            width,
            height,
            camera_count,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT if self.tex_dtype.is_floating_point else gl.GL_UNSIGNED_BYTE,
            None,
        )

        self.batch_depth_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.batch_depth_tex)
        gl.glTexImage3D(
            gl.GL_TEXTURE_2D_ARRAY,
            0,
            gl.GL_DEPTH_COMPONENT24,
            width,
            height,
            camera_count,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_UNSIGNED_INT,
            None,
        )

        self.batch_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.batch_fbo)
        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, self.batch_rgba_tex, 0)
        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, self.batch_depth_tex, 0)
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Incomplete batch framebuffer.')
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
        self.batch_cu_tex = CHECK_CUDART_ERROR(
            cudart.cudaGraphicsGLRegisterImage(self.batch_rgba_tex, gl.GL_TEXTURE_2D_ARRAY, flags)
        )
        self.batch_texture_shape = (camera_count, height, width)

    def init_batch_draw_commands(self, camera_count: int):
        from cuda import cudart

        if hasattr(self, 'batch_cu_draw_commands'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.batch_cu_draw_commands))
            gl.glDeleteBuffers(1, [self.batch_draw_commands])

        self.batch_draw_commands = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_DRAW_INDIRECT_BUFFER, self.batch_draw_commands)
        gl.glBufferData(gl.GL_DRAW_INDIRECT_BUFFER, camera_count * 4 * 4, None, gl.GL_DYNAMIC_DRAW)
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        self.batch_cu_draw_commands = CHECK_CUDART_ERROR(
            cudart.cudaGraphicsGLRegisterBuffer(self.batch_draw_commands, flags)
        )
        self.batch_max_draw_commands = camera_count

    def resize_batch_buffers(self, vertices: int):
        if not hasattr(self, 'batch_max_vertices') or vertices > self.batch_max_vertices:
            self.init_batch_gl_buffers(vertices)

    def resize_batch_draw_commands(self, camera_count: int):
        if not hasattr(self, 'batch_max_draw_commands') or camera_count > self.batch_max_draw_commands:
            self.init_batch_draw_commands(camera_count)

    def resize_batch_textures(self, height: int, width: int, camera_count: int):
        if getattr(self, 'batch_texture_shape', None) != (camera_count, height, width):
            self.init_batch_textures(height, width, camera_count)

    def _prepare_batch(self, vertices, raster_settings):
        height = raster_settings[0].image_height
        width = raster_settings[0].image_width
        self.resize_batch_buffers(vertices)
        self.resize_batch_textures(height, width, len(raster_settings))
        self.use_gl_batch_program()
        self.upload_batch_gl_uniforms(raster_settings)
        return height, width

    def _begin_batch_draw(self, height, width):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.batch_fbo)
        gl.glViewport(0, 0, width, height)
        gl.glScissor(0, 0, width, height)
        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(self.batch_vao)

    def _read_batch_texture(self, camera_count, height, width):
        from cuda import cudart

        stream = torch.cuda.current_stream().cuda_stream
        rgba = torch.empty((camera_count, height, width, 4), dtype=self.tex_dtype, device='cuda')
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.batch_cu_tex, stream))
        for camera_id in range(camera_count):
            texture = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(self.batch_cu_tex, camera_id, 0))
            CHECK_CUDART_ERROR(
                cudart.cudaMemcpy2DFromArrayAsync(
                    rgba[camera_id].data_ptr(),
                    width * 4 * rgba.element_size(),
                    texture,
                    0,
                    0,
                    width * 4 * rgba.element_size(),
                    height,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    stream,
                )
            )
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.batch_cu_tex, stream))
        return rgba

    @staticmethod
    def _cast_batch_output(rgba, dtype):
        if rgba.dtype == dtype:
            return rgba
        if not torch.is_floating_point(rgba):
            return rgba / torch.iinfo(rgba.dtype).max
        return rgba.to(dtype)

    def upload_batch_gl_uniforms(self, raster_settings: Sequence['GaussianRasterizationSettings']):
        camera_count = len(raster_settings)
        if camera_count > 32:
            raise ValueError('BatchedGaussianRasterizer supports at most 32 cameras.')
        first = raster_settings[0]
        if any(
            (setting.use_depth, setting.solid_mode, setting.edge_mode) !=
            (first.use_depth, first.solid_mode, first.edge_mode)
            for setting in raster_settings[1:]
        ):
            raise ValueError('All batch cameras must use the same render flags.')

        projection_values = torch.stack([setting.projmatrix for setting in raster_settings]).detach().cpu().numpy()
        view_values = torch.stack([setting.viewmatrix for setting in raster_settings]).detach().cpu().numpy()
        tanfovs = torch.stack(
            [
                torch.stack(
                    (
                        torch.as_tensor(setting.tanfovx, dtype=torch.float32, device=first.projmatrix.device),
                        torch.as_tensor(setting.tanfovy, dtype=torch.float32, device=first.projmatrix.device),
                    )
                )
                for setting in raster_settings
            ]
        ).detach().cpu().numpy()
        projections = []
        view_matrices = []
        focals = []
        principals = []
        viewports = []
        for camera_id, setting in enumerate(raster_settings):
            projection = mat4(*projection_values[camera_id].tolist())
            view = mat4(*view_values[camera_id].tolist())
            intrinsics = projection * glm.affineInverse(view)
            projections.append(np.ctypeslib.as_array(glm.value_ptr(projection), shape=(16,)).copy())
            view_matrices.append(np.ctypeslib.as_array(glm.value_ptr(view), shape=(16,)).copy())
            focals.append((0.5 * setting.image_width / tanfovs[camera_id, 0], 0.5 * setting.image_height / tanfovs[camera_id, 1]))
            principals.append(((intrinsics[2][0] + 1) * 0.5 * setting.image_width, (intrinsics[2][1] + 1) * 0.5 * setting.image_height))
            viewports.append((1.0 / setting.image_width, 1.0 / setting.image_height))

        gl.glUniform1i(self.batch_uniforms.cameraCount, camera_count)
        gl.glUniformMatrix4fv(self.batch_uniforms.P, camera_count, gl.GL_FALSE, np.concatenate(projections))
        gl.glUniformMatrix4fv(self.batch_uniforms.VM, camera_count, gl.GL_FALSE, np.concatenate(view_matrices))
        gl.glUniform2fv(self.batch_uniforms.focal, camera_count, np.asarray(focals, dtype=np.float32))
        gl.glUniform2fv(self.batch_uniforms.principal, camera_count, np.asarray(principals, dtype=np.float32))
        gl.glUniform2fv(self.batch_uniforms.basisViewport, camera_count, np.asarray(viewports, dtype=np.float32))
        gl.glUniform1i(self.batch_uniforms.useDepth, int(first.use_depth))
        gl.glUniform1i(self.batch_uniforms.solidMode, int(first.solid_mode))
        gl.glUniform1i(self.batch_uniforms.edgeMode, int(first.edge_mode))


    def upload_batch_gl_uniforms(self, raster_settings: Sequence['GaussianRasterizationSettings']):
        camera_count = len(raster_settings)
        if camera_count > 32:
            raise ValueError('BatchedGaussianRasterizer supports at most 32 cameras.')
        first = raster_settings[0]
        if any(
            (setting.use_depth, setting.solid_mode, setting.edge_mode) !=
            (first.use_depth, first.solid_mode, first.edge_mode)
            for setting in raster_settings[1:]
        ):
            raise ValueError('All batch cameras must use the same render flags.')

        projection_values = torch.stack([setting.projmatrix for setting in raster_settings]).detach().cpu().numpy()
        view_values = torch.stack([setting.viewmatrix for setting in raster_settings]).detach().cpu().numpy()
        tanfovs = torch.stack(
            [
                torch.stack(
                    (
                        torch.as_tensor(setting.tanfovx, dtype=torch.float32, device=first.projmatrix.device),
                        torch.as_tensor(setting.tanfovy, dtype=torch.float32, device=first.projmatrix.device),
                    )
                )
                for setting in raster_settings
            ]
        ).detach().cpu().numpy()
        projections = []
        view_matrices = []
        focals = []
        principals = []
        viewports = []
        for camera_id, setting in enumerate(raster_settings):
            projection = mat4(*projection_values[camera_id].tolist())
            view = mat4(*view_values[camera_id].tolist())
            intrinsics = projection * glm.affineInverse(view)
            projections.append(np.ctypeslib.as_array(glm.value_ptr(projection), shape=(16,)).copy())
            view_matrices.append(np.ctypeslib.as_array(glm.value_ptr(view), shape=(16,)).copy())
            focals.append((0.5 * setting.image_width / tanfovs[camera_id, 0], 0.5 * setting.image_height / tanfovs[camera_id, 1]))
            principals.append(((intrinsics[2][0] + 1) * 0.5 * setting.image_width, (intrinsics[2][1] + 1) * 0.5 * setting.image_height))
            viewports.append((1.0 / setting.image_width, 1.0 / setting.image_height))

        gl.glUniform1i(self.batch_uniforms.cameraCount, camera_count)
        gl.glUniformMatrix4fv(self.batch_uniforms.P, camera_count, gl.GL_FALSE, np.concatenate(projections))
        gl.glUniformMatrix4fv(self.batch_uniforms.VM, camera_count, gl.GL_FALSE, np.concatenate(view_matrices))
        gl.glUniform2fv(self.batch_uniforms.focal, camera_count, np.asarray(focals, dtype=np.float32))
        gl.glUniform2fv(self.batch_uniforms.principal, camera_count, np.asarray(principals, dtype=np.float32))
        gl.glUniform2fv(self.batch_uniforms.basisViewport, camera_count, np.asarray(viewports, dtype=np.float32))
        gl.glUniform1i(self.batch_uniforms.useDepth, int(first.use_depth))
        gl.glUniform1i(self.batch_uniforms.solidMode, int(first.solid_mode))
        gl.glUniform1i(self.batch_uniforms.edgeMode, int(first.edge_mode))


    @torch.no_grad()
    def render_batch(
        self,
        means3d: Sequence[torch.Tensor],
        cov6: Sequence[torch.Tensor],
        rgb3: Sequence[torch.Tensor],
        occ1: Sequence[torch.Tensor],
        raster_settings: Sequence['GaussianRasterizationSettings'],
    ):
        if not self.offline_rendering:
            raise RuntimeError('BatchedGaussianRasterizer requires an EGL/offline OpenGL context.')
        if not raster_settings:
            raise ValueError('BatchedGaussianRasterizer requires at least one camera.')
        if self.dtype != torch.float:
            raise RuntimeError('Direct batch VBO packing requires torch.float.')

        height = raster_settings[0].image_height
        width = raster_settings[0].image_width
        if any((setting.image_height, setting.image_width) != (height, width) for setting in raster_settings):
            raise ValueError('All batch cameras must have the same image size.')

        profile = os.environ.get('FAST_GAUSS_BATCH_PROFILE') == '1'
        timings = {}
        if profile:
            torch.cuda.synchronize()
            started_at = time.perf_counter()

        batch_views = []
        vertex_count = 0
        sort_time = 0.0
        for xyz, cov, rgb, occ, setting in zip(means3d, cov6, rgb3, occ1, raster_settings):
            xyz = xyz.to(dtype=self.dtype)
            cov = cov.to(dtype=self.dtype)
            rgb = rgb.to(dtype=self.dtype)
            occ = occ.to(dtype=self.dtype)
            if profile:
                stage_started_at = time.perf_counter()
            view_matrix = torch.as_tensor(setting.viewmatrix, dtype=self.dtype, device=xyz.device)
            view = xyz @ view_matrix[:3, :3] + view_matrix[:3, 3]
            order = view[..., 2].argsort(descending=True)
            if profile:
                torch.cuda.synchronize()
                sort_time += time.perf_counter() - stage_started_at
            batch_views.append((xyz, cov, rgb, occ, order))
            vertex_count += len(order)

        if profile:
            torch.cuda.synchronize()
            timings['view_and_sort'] = sort_time
            timings['sort_and_pack'] = time.perf_counter() - started_at
            started_at = time.perf_counter()
        height, width = self._prepare_batch(vertex_count, raster_settings)
        if profile:
            timings['gl_setup_and_uniforms'] = time.perf_counter() - started_at
            started_at = time.perf_counter()

        from cuda import cudart

        stream = torch.cuda.current_stream().cuda_stream
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.batch_cu_vbo, stream))
        vbo_ptr, _ = CHECK_CUDART_ERROR(cudart.cudaGraphicsResourceGetMappedPointer(self.batch_cu_vbo))
        if profile:
            torch.cuda.synchronize()
            timings['cuda_gl_vbo_map'] = time.perf_counter() - started_at
            started_at = time.perf_counter()
        offset = 0
        for xyz, cov, rgb, occ, order in batch_views:
            _pack_batch_vbo(vbo_ptr + offset * 13 * 4, xyz, cov, rgb, occ, order, stream)
            offset += len(order)
        if profile:
            torch.cuda.synchronize()
            timings['cuda_vbo_pack'] = time.perf_counter() - started_at
            started_at = time.perf_counter()
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.batch_cu_vbo, stream))
        if profile:
            torch.cuda.synchronize()
            timings['cuda_gl_vbo_unmap'] = time.perf_counter() - started_at
            started_at = time.perf_counter()

        self._begin_batch_draw(height, width)
        draw_offsets = np.cumsum([0] + [len(view[4]) for view in batch_views[:-1]], dtype=np.int32)
        draw_counts = np.asarray([len(view[4]) for view in batch_views], dtype=np.int32)
        gl.glMultiDrawArrays(gl.GL_POINTS, draw_offsets, draw_counts, len(batch_views))
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

        return self._cast_batch_output(rgba, means3d[0].dtype)


    def rasterize_gaussians_batch(
        self,
        means3d: torch.Tensor,
        means2d: torch.Tensor,
        shs: torch.Tensor,
        colors_precomp: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3d_precomp: torch.Tensor,
        raster_settings: Sequence['GaussianRasterizationSettings'],
    ):
        if cov3d_precomp is None:
            cov3d_precomp = build_cov6(scales, rotations)

        profile = os.environ.get('FAST_GAUSS_BATCH_PROFILE') == '1'
        preprocessing_timings = {}
        if profile:
            torch.cuda.synchronize()
            started_at = time.perf_counter()
        frustum_time = 0.0
        geometry_gather_time = 0.0
        color_gather_time = 0.0
        per_camera_means = []
        per_camera_covariances = []
        per_camera_colors = []
        per_camera_opacities = []
        homogeneous_means = None
        if any(not setting.prefiltered for setting in raster_settings):
            homogeneous_means = torch.cat([means3d, torch.ones_like(means3d[..., :1])], dim=-1)
        for camera_id, setting in enumerate(raster_settings):
            if setting.prefiltered:
                visible = None
            else:
                if profile:
                    stage_started_at = time.perf_counter()
                ndc = homogeneous_means @ torch.as_tensor(setting.projmatrix, device=means3d.device)
                ndc = ndc[..., :3] / ndc[..., 3:]
                visible = (
                    (ndc[..., 2] > -1.01) & (ndc[..., 2] < 1.01) &
                    (ndc[..., 0] > -1.5) & (ndc[..., 0] < 1.5) &
                    (ndc[..., 1] > -1.5) & (ndc[..., 1] < 1.5)
                ).nonzero()[..., 0]
                if profile:
                    torch.cuda.synchronize()
                    frustum_time += time.perf_counter() - stage_started_at

            if profile:
                stage_started_at = time.perf_counter()
            if visible is None:
                camera_means = means3d
                camera_covariances = cov3d_precomp
                camera_opacities = opacities
            else:
                camera_means = means3d[visible]
                camera_covariances = cov3d_precomp[visible]
                camera_opacities = opacities[visible]
            if profile:
                torch.cuda.synchronize()
                geometry_gather_time += time.perf_counter() - stage_started_at

            if profile:
                stage_started_at = time.perf_counter()
            if colors_precomp is None:
                camera_center = torch.as_tensor(setting.campos, device=means3d.device)
                colors = eval_sh(
                    setting.sh_degree,
                    (shs if visible is None else shs[visible]).mT,
                    normalize(camera_means - camera_center),
                )
                colors = (colors + 0.5).clip(0, 1)
            elif colors_precomp.ndim == 3:
                colors = colors_precomp[camera_id]
                colors = colors if visible is None else colors[visible]
            else:
                colors = colors_precomp if visible is None else colors_precomp[visible]
            if profile:
                torch.cuda.synchronize()
                color_gather_time += time.perf_counter() - stage_started_at

            per_camera_means.append(camera_means)
            per_camera_covariances.append(camera_covariances)
            per_camera_colors.append(colors)
            per_camera_opacities.append(camera_opacities)

        if profile:
            preprocessing_timings['batch_preprocess_total'] = time.perf_counter() - started_at
            preprocessing_timings['batch_frustum'] = frustum_time
            preprocessing_timings['batch_geometry_gather'] = geometry_gather_time
            preprocessing_timings['batch_color_gather'] = color_gather_time
        rgba = self.render_batch(
            per_camera_means,
            per_camera_covariances,
            per_camera_colors,
            per_camera_opacities,
            raster_settings,
        )
        if profile:
            self.last_batch_profile = {**preprocessing_timings, **self.last_batch_profile}
        image, alpha = rgba[..., :3].permute(0, 3, 1, 2), rgba[..., 3:].permute(0, 3, 1, 2)
        return image.float(), alpha.float()
