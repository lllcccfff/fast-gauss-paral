from typing import List, Dict, Optional, Sequence, TYPE_CHECKING
import os
import time

if TYPE_CHECKING:
    from . import GaussianRasterizationSettings

import glm
import math
import torch
import ctypes
import numpy as np

from glm import mat4

from .console_utils import *
from .sh_utils import eval_sh
from .cuda_utils import CHECK_CUDART_ERROR
from .math_utils import normalize, point_padding
from .gl_utils import load_shader_source, use_gl_program
from .gaussian_utils import build_cov6, in_frustum

import OpenGL.GL as gl
from OpenGL.GL import shaders


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


class GSplatContextManager:
    """
    This is a high performance rendering backend of the gaussian splatting paper
    We use CUDA to perform primitive sorting and then render the splats using geometry shader that emits quads
    There's some cuda-gl interop involved, thus cuda-python is required, wsl2 out of the question

    EGL should be automatically supported by this and thus it should be able to be used in an offline environment
    """

    def __init__(self,
                 init_buffer_size: int = 32768,
                 init_texture_size: List[int] = [512, 512],
                 dtype: str = torch.float,  # +5-10 fps on 3060, not working too well with flame_salmon
                 tex_dtype: str = torch.uint8,  # +5-10 fps on 3060, not working too well with flame_salmon

                 # DEBUG: whether to write back to cuda in offline rendering mode, only used for speed tests
                 offline_writeback: bool = True,
                 ):
        self.attr_sizes = [3, 3, 3, 4]  # verts, cov6, rgba4
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.tex_dtype = getattr(torch, tex_dtype) if isinstance(tex_dtype, str) else tex_dtype
        self.gl_tex_dtype = gl.GL_RGBA16F if self.tex_dtype == torch.half else gl.GL_RGBA32F if self.tex_dtype == torch.float else gl.GL_RGBA8
        self.gl_attr_dtypes = [gl.GL_FLOAT if self.dtype == torch.float else gl.GL_HALF_FLOAT] * len(self.attr_sizes)
        self.uniforms = dotdict()  # uniform values

        # Perform actual rendering
        from .gl_utils import eglctx
        self.offline_rendering = eglctx is not None
        self.offline_writeback = offline_writeback

        self.compile_shaders()
        self.use_gl_program(self.gsplat_program)
        self.opengl_options()
        self.resize_buffers(init_buffer_size)
        self.resize_textures(*init_texture_size)

        log(bold(f'[FAST GAUSS] GSplatContextManager initialized with attribute dtype: {self.dtype}, texture dtype: {self.tex_dtype}, offline rendering: {self.offline_rendering}, vertex buffer size: {init_buffer_size}, render buffer size: {init_texture_size}'))

        if not self.offline_rendering:
            log(bold('[FAST GAUSS] Using online rendering mode, in this mode, calling the rendering function of fast_gauss will write directly to the currently bound framebuffer'))
            log(bold('[FAST GAUSS] In this mode, the output of all rasterization calls will be None (same output count). Please do not perform further processing on them'))
            log(bold('[FAST GAUSS] Please make sure to set up the correct GUI environment before calling the rasterization function, see more in readme.md'))

    def opengl_options(self):
        # Performs face culling
        gl.glDisable(gl.GL_CULL_FACE)

        # Performs z-buffer testing
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_ALPHA_TEST)

        # Enable some masking tests
        gl.glEnable(gl.GL_SCISSOR_TEST)

        # Enable blending for final rendering
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFuncSeparate(
            gl.GL_SRC_ALPHA,
            gl.GL_ONE_MINUS_SRC_ALPHA,
            gl.GL_ONE,
            gl.GL_ONE_MINUS_SRC_ALPHA,
        )

    def compile_shaders(self):
        try:
            self.gsplat_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('dsplat.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('dsplat.geom'), gl.GL_GEOMETRY_SHADER),
                shaders.compileShader(load_shader_source('dsplat.frag'), gl.GL_FRAGMENT_SHADER)
            )
            self.gsplat_batch_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('dsplat_batch_drawid.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('dsplat_batch_drawid.geom'), gl.GL_GEOMETRY_SHADER),
                shaders.compileShader(load_shader_source('dsplat.frag'), gl.GL_FRAGMENT_SHADER),
            )
        except Exception as e:
            print(str(e).encode('utf-8').decode('unicode_escape'))
            raise e

    def use_gl_program(self, program: shaders.ShaderProgram):
        use_gl_program(program)
        self.uniforms.P = gl.glGetUniformLocation(program, "P")
        self.uniforms.VM = gl.glGetUniformLocation(program, "VM")
        self.uniforms.focal = gl.glGetUniformLocation(program, "focal")
        self.uniforms.principal = gl.glGetUniformLocation(program, "principal")
        self.uniforms.basisViewport = gl.glGetUniformLocation(program, "basisViewport")
        self.uniforms.useDepth = gl.glGetUniformLocation(program, "useDepth")
        self.uniforms.solidMode = gl.glGetUniformLocation(program, "solidMode")
        self.uniforms.edgeMode = gl.glGetUniformLocation(program, "edgeMode")

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

    def upload_gl_uniforms(self, raster_settings: 'GaussianRasterizationSettings'):
        # FIXME: Possible nasty synchronization issue: raster_settings might contain cuda tensors
        # TODO: Add a warning here to make the user pass in cpu tensors to avoid explicit synchronization

        P = raster_settings.projmatrix.detach().cpu().numpy() if isinstance(raster_settings.projmatrix, torch.Tensor) else raster_settings.projmatrix  # 4,4 # MARK: possible sync
        P = mat4(*P.tolist())
        V = raster_settings.viewmatrix.detach().cpu().numpy() if isinstance(raster_settings.viewmatrix, torch.Tensor) else raster_settings.viewmatrix  # 4,4 # MARK: possible sync
        V = mat4(*V.tolist())

        K = P * glm.affineInverse(V)
        cx = (K[2][0] + 1) / 2 * raster_settings.image_width
        cy = (K[2][1] + 1) / 2 * raster_settings.image_height
        if not self.offline_rendering:

            gl_c2w = glm.affineInverse(V)
            gl_c2w[0] *= 1  # do notflip x
            gl_c2w[1] *= -1  # flip y
            gl_c2w[2] *= -1  # flip z
            V = glm.affineInverse(gl_c2w)

            K[2][0] *= -1
            K[2][2] *= -1
            K[2][3] *= -1

            P = K * V

        M = glm.identity(mat4)
        VM = V * M

        gl.glUniformMatrix4fv(self.uniforms.P, 1, gl.GL_FALSE, glm.value_ptr(P))  # o2c
        gl.glUniformMatrix4fv(self.uniforms.VM, 1, gl.GL_FALSE, glm.value_ptr(VM))  # o2w
        gl.glUniform2f(self.uniforms.focal, 0.5 * raster_settings.image_width / raster_settings.tanfovx, 0.5 * raster_settings.image_height / raster_settings.tanfovy)  # focal in pixel space
        gl.glUniform2f(self.uniforms.principal, cx, cy)  # focal
        gl.glUniform2f(self.uniforms.basisViewport, 1 / raster_settings.image_width, 1 / raster_settings.image_height)  # focal
        gl.glUniform1i(self.uniforms.useDepth, 1 if raster_settings.use_depth else 0)
        gl.glUniform1i(self.uniforms.solidMode, 1 if raster_settings.solid_mode else 0)
        gl.glUniform1i(self.uniforms.edgeMode, 1 if raster_settings.edge_mode else 0)

    def init_gl_buffers(self, v: int = 0):
        from cuda import cudart
        if hasattr(self, 'cu_vbo'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_vbo))

        # This will only init the corresponding buffer object
        element_size = 4 if self.gl_attr_dtypes[0] == gl.GL_FLOAT else 2
        attr_size = sum(self.attr_sizes)
        n_verts_bytes = v * attr_size * element_size

        # Housekeeping
        if hasattr(self, 'vao'):
            gl.glDeleteVertexArrays(1, [self.vao])
            gl.glDeleteBuffers(2, [self.vbo, self.ebo])

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, n_verts_bytes, ctypes.c_void_p(0), gl.GL_DYNAMIC_DRAW)  # NOTE: Using pointers here won't work

        # https://stackoverflow.com/questions/67195932/pyopengl-cannot-render-any-vao
        cumsum = 0
        for i, (s, t) in enumerate(zip(self.attr_sizes, self.gl_attr_dtypes)):
            gl.glVertexAttribPointer(i, s, t, gl.GL_FALSE, attr_size * element_size, ctypes.c_void_p(cumsum * element_size))  # we use 32 bit float
            gl.glEnableVertexAttribArray(i)
            cumsum += s
        # Register vertex buffer obejct
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        try:
            self.cu_vbo = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterBuffer(self.vbo, flags))
        except RuntimeError as e:
            log(red(f'Your system does not support CUDA-GL interop, will use pytorch3d\'s implementation instead'))
            log(red(f'This can be done by specifying {blue("model_cfg.sampler_cfg.use_cudagl=False model_cfg.sampler_cfg.use_diffgl=False")} at the end of your command'))
            log(red(f'Note that this implementation is extremely slow, we recommend running on a native system that support the interop'))
            log(red(f'An alternative is to install diff_point_rasterization and use the approximated tile-based rasterization, enabled by the `render_gs` switch'))
            # raise RuntimeError(str(e) + ": This unrecoverable, please read the error message above")
            raise e

    def init_textures(self, H: int, W: int):
        from cuda import cudart
        if hasattr(self, 'cu_tex'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_tex))

        if hasattr(self, 'fbo'):
            gl.glDeleteFramebuffers(1, [self.fbo])
            gl.glDeleteRenderbuffers(1, [self.rbo_rgba, self.rbo_atth])

        # Prepare for write frame buffers
        self.rbo_rgba = gl.glGenRenderbuffers(1)
        self.rbo_atth = gl.glGenRenderbuffers(1)
        self.fbo = gl.glGenFramebuffers(1)  # generate 1 framebuffer, storereference in fb

        # Init the texture (call the resizing function), will simply allocate empty memory
        # The internal format describes how the texture shall be stored in the GPU. The format describes how the format of your pixel data in client memory (together with the type parameter).
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo_rgba)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, self.gl_tex_dtype, W, H)  # faster if using rgba8
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo_atth)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, W, H)

        # Bind texture to fbo
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, self.rbo_rgba)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self.rbo_atth)
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])

        # Check framebuffer status
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            log(red('Framebuffer not complete, exiting...'))
            raise RuntimeError('Incomplete framebuffer')

        # Restore the original state
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Register image to read from
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
        self.cu_tex = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.rbo_rgba, gl.GL_RENDERBUFFER, flags))

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

    def resize_batch_buffers(self, vertices: int):
        if not hasattr(self, 'batch_max_vertices') or vertices > self.batch_max_vertices:
            self.init_batch_gl_buffers(vertices)

    def resize_batch_textures(self, height: int, width: int, camera_count: int):
        if getattr(self, 'batch_texture_shape', None) != (camera_count, height, width):
            self.init_batch_textures(height, width, camera_count)

    @staticmethod
    def _matrix_array(matrix):
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
        return np.asarray(matrix, dtype=np.float32)

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

        projections = []
        view_matrices = []
        focals = []
        principals = []
        viewports = []
        for setting in raster_settings:
            projection = mat4(*self._matrix_array(setting.projmatrix).tolist())
            view = mat4(*self._matrix_array(setting.viewmatrix).tolist())
            intrinsics = projection * glm.affineInverse(view)
            projections.append(np.ctypeslib.as_array(glm.value_ptr(projection), shape=(16,)).copy())
            view_matrices.append(np.ctypeslib.as_array(glm.value_ptr(view), shape=(16,)).copy())
            focals.append((0.5 * setting.image_width / setting.tanfovx, 0.5 * setting.image_height / setting.tanfovy))
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

    def resize_textures(self, H: int, W: int):  # analogy to update_gl_buffers
        init = False
        if not hasattr(self, 'max_H'): self.max_H = 0; init = True
        if not hasattr(self, 'max_W'): self.max_W = 0; init = True
        if H > self.max_H or W > self.max_W:  # max got updated
            if H > self.max_H: self.max_H = int(H * 1.05)
            if W > self.max_W: self.max_W = int(W * 1.05)
            if not init: log(bold('[FAST GAUSS] Resizing render buffers to:'), int(H), int(W))
            self.init_textures(self.max_H, self.max_W)

    def resize_buffers(self, v: int = 0):
        init = False
        if not hasattr(self, 'max_verts'): self.max_verts = 0; init = True
        if v > self.max_verts:
            if v > self.max_verts: self.max_verts = int(v * 1.05)
            if not init: log(bold('[FAST GAUSS] Resizing vertex buffers to:'), int(v))
            self.init_gl_buffers(self.max_verts)

    @torch.no_grad()
    def render(self, xyz3: torch.Tensor, cov6: torch.Tensor, rgb3: torch.Tensor, occ1: torch.Tensor, raster_settings: 'GaussianRasterizationSettings'):
        # TODO: Faster way to do this?
        self.use_gl_program(self.gsplat_program)

        # Compute GS
        data = torch.cat([xyz3, cov6, rgb3, occ1], dim=-1)  # slow memory operation
        stream = torch.cuda.current_stream().cuda_stream

        # Preparing dtype
        if data.dtype != self.dtype:
            warn_once(yellow(f'Input tensors has dtype {data.dtype}, expected {self.dtype}, will cast to {self.dtype}'))
            data = data.type(self.dtype)
            xyz3 = xyz3.type(self.dtype)
            raster_settings = dotdict(raster_settings._asdict())
            for key in raster_settings:
                if isinstance(raster_settings[key], torch.Tensor):
                    raster_settings[key] = raster_settings[key].type(self.dtype)
                elif isinstance(raster_settings[key], np.ndarray):
                    raster_settings[key] = torch.from_numpy(raster_settings[key]).type(self.dtype).numpy()  # HACK: More efficient way?

        # Prepare OpenGL texture size
        render_h, render_w = raster_settings.image_height, raster_settings.image_width
        self.resize_textures(render_h, render_w)
        self.resize_buffers(len(xyz3))

        # Sort by view space depth
        # FIXME: For unknown reasons, sorting also required for solid mode?
        # if not raster_settings.solid_mode:  # only perform sorting when not using solid mode
        # view = point_padding(xyz3) @ raster_settings.viewmatrix.to('cuda', non_blocking=True)
        w2cT = torch.as_tensor(raster_settings.viewmatrix).to(data.device, non_blocking=True)
        view = xyz3 @ w2cT[:3, :3] + w2cT[:3, 3]
        idx = view[..., 2].argsort(descending=True)  # S, sorted
        data = data[idx].ravel().contiguous()  # sorted data on gpu

        # Upload sorted data to OpenGL for rendering
        from cuda import cudart
        from .cuda_utils import CHECK_CUDART_ERROR, FORMAT_CUDART_ERROR
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.cu_vbo, stream))
        cu_vbo_ptr, cu_vbo_size = CHECK_CUDART_ERROR(cudart.cudaGraphicsResourceGetMappedPointer(self.cu_vbo))

        CHECK_CUDART_ERROR(cudart.cudaMemcpyAsync(cu_vbo_ptr,
                                                  data.data_ptr(),
                                                  data.numel() * data.element_size(),
                                                  kind,
                                                  stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.cu_vbo, stream))

        fb_x, fb_y, fb_w, fb_h = gl.glGetIntegerv(gl.GL_VIEWPORT)
        if self.offline_rendering or not (render_w == fb_w and render_h == fb_h and fb_x == 0 and fb_y == 0):
            # Offline rendering or rendering to a differenly sized framebuffer
            # Will render to the texture and then perform a blit
            old = gl.glGetInteger(gl.GL_DRAW_FRAMEBUFFER_BINDING)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)  # for offscreen rendering to textures
            gl.glClearColor(0, 0, 0, 0.)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if not (render_w == fb_w and render_h == fb_h and fb_x == 0 and fb_y == 0):
            # Change the viewport and scissor test region
            gl.glViewport(0, 0, render_w, render_h)
            gl.glScissor(0, 0, render_w, render_h)  # only render in this small region of the viewport

        # Perform the actual rendering
        # Will simply render to the currently bound framebuffer
        self.upload_gl_uniforms(raster_settings)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(xyz3))  # number of vertices
        gl.glBindVertexArray(0)

        if not self.offline_rendering and not (render_w == fb_w and render_h == fb_h and fb_x == 0 and fb_y == 0):
            # Restore the original framebuffer and viewport
            gl.glViewport(fb_x, fb_y, fb_w, fb_h)
            gl.glScissor(fb_x, fb_y, fb_w, fb_h)  # only render in this small region of the viewport

            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, old)  # read buffer defaults to 0
            gl.glBlitFramebuffer(0, 0, render_w, render_h,
                                 fb_x, fb_y, fb_x + fb_w, fb_y + fb_h,
                                 gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)  # now self.tex contains the content of the already rendered frame
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, old)

        # TODO: Implement rendering normal as the gradient on screen space depth
        # Need to use textures instead of renderbuffers?
        # Should we do this in CUDA or just in plain OpenGL?
        # If in CUDA, need to perform some copy operations, but renderbuffers should be fine
        # If in OpenGL, we need to divise a full screen quad and perform the gradient computation using a texture in the fragment shader

        # Early return if in online rendering mode
        if not self.offline_rendering and self.offline_writeback:
            return None

        # Prepare the output for offline rendering
        rgba_map = torch.empty((render_h, render_w, 4), dtype=self.tex_dtype, device='cuda')  # to hold the data from opengl
        # Texture access and copy could be very slow...
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Copy rendered image and depth back as tensor
        cu_tex = self.cu_tex

        # The resources in resources may be accessed by CUDA until they are unmapped.
        # The graphics API from which resources were registered should not access any resources while they are mapped by CUDA.
        # If an application does so, the results are undefined.
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, cu_tex, stream))
        cu_tex_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(cu_tex, 0, 0))
        CHECK_CUDART_ERROR(cudart.cudaMemcpy2DFromArrayAsync(rgba_map.data_ptr(),  # dst
                                                             render_w * 4 * rgba_map.element_size(),  # dpitch
                                                             cu_tex_arr,  # src
                                                             0,  # wOffset
                                                             0,  # hOffset
                                                             render_w * 4 * rgba_map.element_size(),  # width Width of matrix transfer (columns in bytes)
                                                             render_h,  # height
                                                             kind,  # kind
                                                             stream))  # stream
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, cu_tex, stream))

        if rgba_map.dtype != xyz3.dtype:
            warn_once(yellow(f'[FAST GAUSS] Using render buffer dtype {rgba_map.dtype}, expected {xyz3.dtype} for the output, will cast to {xyz3.dtype}'))
            if not torch.is_floating_point(rgba_map):
                warn_once(yellow(f'[FAST GAUSS] Using a non-floating-point render buffer dtype: {rgba_map.dtype}, this might cause some precision loss'))
                rgba_map = rgba_map / torch.iinfo(rgba_map.dtype).max  # should be 255 for uint8
            else:
                rgba_map = rgba_map.to(xyz3.dtype)

        return rgba_map  # render_h, render_w, 4

    def rasterize_gaussians(
        self,
        means3D: torch.Tensor,  # N, 3
        means2D: torch.Tensor,  # N, 2
        shs: torch.Tensor,  # N, SH, 3
        colors_precomp: torch.Tensor,  # N, 3
        opacities: torch.Tensor,  # N, 1
        scales: torch.Tensor,  # N, 3
        rotations: torch.Tensor,  # N, 4
        cov3D_precomp: torch.Tensor,  # N, 6
        raster_settings: 'GaussianRasterizationSettings',
    ):
        # FIXME: This preprocessing stage is extremely slow, need to optimize this to raw cuda or write it in the shader
        if raster_settings.prefiltered:

            if cov3D_precomp is None:
                cov3D_precomp = build_cov6(scales, rotations)  # N, 6

            if colors_precomp is None:
                C = torch.as_tensor(raster_settings.campos).to('cuda', non_blocking=True)  # 3,
                dirs = normalize(means3D - C)
                colors_precomp = eval_sh(raster_settings.sh_degree, shs.mT, dirs)
                colors_precomp = (colors_precomp + 0.5).clip(0, 1)

        else:

            visible = in_frustum(means3D, torch.as_tensor(raster_settings.projmatrix).to('cuda', non_blocking=True)).nonzero()[..., 0]  # S, # MARK: SYNC
            means3D = means3D[visible]
            opacities = opacities[visible]

            if cov3D_precomp is None:
                cov3D_precomp = build_cov6(scales[visible], rotations[visible])
            else:
                cov3D_precomp = cov3D_precomp[visible]

            if colors_precomp is None:
                C = torch.as_tensor(raster_settings.campos).to('cuda', non_blocking=True)  # 3,
                dirs = normalize(means3D - C)
                colors_precomp = eval_sh(raster_settings.sh_degree, shs[visible].mT, dirs)
                colors_precomp = (colors_precomp + 0.5).clip(0, 1)
            else:
                colors_precomp = colors_precomp[visible]

        rgba_map = self.render(means3D, cov3D_precomp, colors_precomp, opacities, raster_settings)

        if self.offline_rendering and self.offline_writeback:
            image, alpha = rgba_map.split([3, 1], dim=-1)
            image, alpha = image.permute(2, 0, 1), alpha.permute(2, 0, 1)

            return image.float(), alpha.float()
        else:
            return None, None

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
        self.resize_batch_buffers(vertex_count)
        self.resize_batch_textures(height, width, len(raster_settings))
        self.use_gl_batch_program()
        self.upload_batch_gl_uniforms(raster_settings)
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

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.batch_fbo)
        gl.glViewport(0, 0, width, height)
        gl.glScissor(0, 0, width, height)
        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(self.batch_vao)
        draw_offsets = np.cumsum([0] + [len(view[4]) for view in batch_views[:-1]], dtype=np.int32)
        draw_counts = np.asarray([len(view[4]) for view in batch_views], dtype=np.int32)
        gl.glMultiDrawArrays(gl.GL_POINTS, draw_offsets, draw_counts, len(batch_views))
        gl.glBindVertexArray(0)
        if profile:
            gl.glFinish()
            timings['layered_draw'] = time.perf_counter() - started_at
            started_at = time.perf_counter()

        rgba = torch.empty((len(raster_settings), height, width, 4), dtype=self.tex_dtype, device='cuda')
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.batch_cu_tex, stream))
        for camera_id in range(len(raster_settings)):
            tex_layer = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(self.batch_cu_tex, camera_id, 0))
            CHECK_CUDART_ERROR(
                cudart.cudaMemcpy2DFromArrayAsync(
                    rgba[camera_id].data_ptr(),
                    width * 4 * rgba.element_size(),
                    tex_layer,
                    0,
                    0,
                    width * 4 * rgba.element_size(),
                    height,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    stream,
                )
            )
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.batch_cu_tex, stream))
        if profile:
            torch.cuda.synchronize()
            timings['cuda_gl_readback'] = time.perf_counter() - started_at
            self.last_batch_profile = timings

        if rgba.dtype != means3d[0].dtype:
            if not torch.is_floating_point(rgba):
                rgba = rgba / torch.iinfo(rgba.dtype).max
            else:
                rgba = rgba.to(means3d[0].dtype)
        return rgba

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
