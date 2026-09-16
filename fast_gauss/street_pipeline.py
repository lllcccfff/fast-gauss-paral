"""Fused CUDA preparation and draw packing for StreetWorld Gaussians."""

import ctypes
import os

import torch


_MODULE = None
_KERNELS = None
last_profile = {}


_SOURCE = r'''
#define PI 3.14159265358979323846f
#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

struct V3 { float x; float y; float z; };

__device__ __forceinline__ V3 v3(float x, float y, float z) { return {x, y, z}; }
__device__ __forceinline__ V3 add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ V3 mul(V3 a, float b) { return v3(a.x * b, a.y * b, a.z * b); }
__device__ __forceinline__ V3 read3(const float* values, int index) {
    const float* value = values + 3 * index;
    return v3(value[0], value[1], value[2]);
}
__device__ __forceinline__ void write3(float* values, int index, V3 value) {
    float* output = values + 3 * index;
    output[0] = value.x; output[1] = value.y; output[2] = value.z;
}

__device__ __forceinline__ void quaternion_rotation(const float* quaternion, float* rotation) {
    const float w = quaternion[0];
    const float x = quaternion[1];
    const float y = quaternion[2];
    const float z = quaternion[3];
    rotation[0] = 1.f - 2.f * (y * y + z * z);
    rotation[1] = 2.f * (x * y - w * z);
    rotation[2] = 2.f * (x * z + w * y);
    rotation[3] = 2.f * (x * y + w * z);
    rotation[4] = 1.f - 2.f * (x * x + z * z);
    rotation[5] = 2.f * (y * z - w * x);
    rotation[6] = 2.f * (x * z - w * y);
    rotation[7] = 2.f * (y * z + w * x);
    rotation[8] = 1.f - 2.f * (x * x + y * y);
}

__device__ __forceinline__ void covariance_from_rotation(const float* scale, const float* rotation, float* covariance) {
    const float sx = scale[0] * scale[0];
    const float sy = scale[1] * scale[1];
    const float sz = scale[2] * scale[2];
    covariance[0] = rotation[0] * rotation[0] * sx + rotation[1] * rotation[1] * sy + rotation[2] * rotation[2] * sz;
    covariance[1] = rotation[0] * rotation[3] * sx + rotation[1] * rotation[4] * sy + rotation[2] * rotation[5] * sz;
    covariance[2] = rotation[0] * rotation[6] * sx + rotation[1] * rotation[7] * sy + rotation[2] * rotation[8] * sz;
    covariance[3] = rotation[3] * rotation[3] * sx + rotation[4] * rotation[4] * sy + rotation[5] * rotation[5] * sz;
    covariance[4] = rotation[3] * rotation[6] * sx + rotation[4] * rotation[7] * sy + rotation[5] * rotation[8] * sz;
    covariance[5] = rotation[6] * rotation[6] * sx + rotation[7] * rotation[7] * sy + rotation[8] * rotation[8] * sz;
}

__device__ __forceinline__ void rotate_covariance(const float* pose, const float* covariance, float* output) {
    const float rotation[9] = {
        pose[0], pose[1], pose[2],
        pose[4], pose[5], pose[6],
        pose[8], pose[9], pose[10],
    };
    float sigma[9] = {
        covariance[0], covariance[1], covariance[2],
        covariance[1], covariance[3], covariance[4],
        covariance[2], covariance[4], covariance[5],
    };
    float temp[9];
    float result[9];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            temp[3 * row + column] = 0.f;
            for (int inner = 0; inner < 3; ++inner) {
                temp[3 * row + column] += rotation[3 * row + inner] * sigma[3 * inner + column];
            }
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result[3 * row + column] = 0.f;
            for (int inner = 0; inner < 3; ++inner) {
                result[3 * row + column] += temp[3 * row + inner] * rotation[3 * column + inner];
            }
        }
    }
    output[0] = result[0]; output[1] = result[1]; output[2] = result[2];
    output[3] = result[4]; output[4] = result[5]; output[5] = result[8];
}

__device__ __forceinline__ V3 transform_point(const float* pose, V3 point) {
    return v3(
        pose[0] * point.x + pose[1] * point.y + pose[2] * point.z + pose[3],
        pose[4] * point.x + pose[5] * point.y + pose[6] * point.z + pose[7],
        pose[8] * point.x + pose[9] * point.y + pose[10] * point.z + pose[11]
    );
}

__device__ __forceinline__ void conditional_covariance(
    const float* scale, float scale_t, const float* quaternion, const float* quaternion_r,
    float* covariance, float* covariance_t, V3* speed
) {
    float left[16] = {
        quaternion[0], quaternion[1], -quaternion[2], quaternion[3],
        -quaternion[1], quaternion[0], quaternion[3], quaternion[2],
        quaternion[2], -quaternion[3], quaternion[0], quaternion[1],
        -quaternion[3], -quaternion[2], -quaternion[1], quaternion[0],
    };
    float right[16] = {
        quaternion_r[0], quaternion_r[1], -quaternion_r[2], -quaternion_r[3],
        -quaternion_r[1], quaternion_r[0], quaternion_r[3], -quaternion_r[2],
        quaternion_r[2], -quaternion_r[3], quaternion_r[0], -quaternion_r[1],
        quaternion_r[3], quaternion_r[2], quaternion_r[1], quaternion_r[0],
    };
    float rotation[16];
    float transformed[16];
    float sigma[16];
    const float scales[4] = {scale[0], scale[1], scale[2], scale_t};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            rotation[4 * row + column] = 0.f;
            for (int inner = 0; inner < 4; ++inner) {
                rotation[4 * row + column] += right[4 * row + inner] * left[4 * inner + column];
            }
            transformed[4 * row + column] = rotation[4 * row + column] * scales[column];
        }
    }
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            sigma[4 * row + column] = 0.f;
            for (int inner = 0; inner < 4; ++inner) {
                sigma[4 * row + column] += transformed[4 * row + inner] * transformed[4 * column + inner];
            }
        }
    }
    const float cov_t = sigma[15];
    const float cov12x = sigma[3];
    const float cov12y = sigma[7];
    const float cov12z = sigma[11];
    covariance[0] = sigma[0] - cov12x * cov12x / cov_t;
    covariance[1] = sigma[1] - cov12x * cov12y / cov_t;
    covariance[2] = sigma[2] - cov12x * cov12z / cov_t;
    covariance[3] = sigma[5] - cov12y * cov12y / cov_t;
    covariance[4] = sigma[6] - cov12y * cov12z / cov_t;
    covariance[5] = sigma[10] - cov12z * cov12z / cov_t;
    *covariance_t = cov_t;
    *speed = v3(cov12x / cov_t, cov12y / cov_t, cov12z / cov_t);
}

__device__ __forceinline__ V3 feature(const float* features, int index, int stride, int coefficient) {
    return read3(features, index * stride + coefficient);
}

struct SHBasis { float x; float y; float z; float xx; float yy; float zz; };

__device__ __forceinline__ SHBasis sh_basis(V3 direction) {
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    return {x, y, z, x * x, y * y, z * z};
}

__device__ __forceinline__ float sh0(SHBasis b) { return SH_C0; }
__device__ __forceinline__ float sh1(SHBasis b) { return -SH_C1 * b.y; }
__device__ __forceinline__ float sh2(SHBasis b) { return SH_C1 * b.z; }
__device__ __forceinline__ float sh3(SHBasis b) { return -SH_C1 * b.x; }
__device__ __forceinline__ float sh4(SHBasis b) { return 1.0925484305920792f * b.x * b.y; }
__device__ __forceinline__ float sh5(SHBasis b) { return -1.0925484305920792f * b.y * b.z; }
__device__ __forceinline__ float sh6(SHBasis b) { return .31539156525252005f * (2.f * b.zz - b.xx - b.yy); }
__device__ __forceinline__ float sh7(SHBasis b) { return -1.0925484305920792f * b.x * b.z; }
__device__ __forceinline__ float sh8(SHBasis b) { return .5462742152960396f * (b.xx - b.yy); }
__device__ __forceinline__ float sh9(SHBasis b) { return -.5900435899266435f * b.y * (3.f * b.xx - b.yy); }
__device__ __forceinline__ float sh10(SHBasis b) { return 2.890611442640554f * b.x * b.y * b.z; }
__device__ __forceinline__ float sh11(SHBasis b) { return -.4570457994644658f * b.y * (4.f * b.zz - b.xx - b.yy); }
__device__ __forceinline__ float sh12(SHBasis b) { return .3731763325901154f * b.z * (2.f * b.zz - 3.f * b.xx - 3.f * b.yy); }
__device__ __forceinline__ float sh13(SHBasis b) { return -.4570457994644658f * b.x * (4.f * b.zz - b.xx - b.yy); }
__device__ __forceinline__ float sh14(SHBasis b) { return 1.445305721320277f * b.z * (b.xx - b.yy); }
__device__ __forceinline__ float sh15(SHBasis b) { return -.5900435899266435f * b.x * (b.xx - 3.f * b.yy); }

__device__ __forceinline__ V3 add_feature(
    V3 result, const float* features, int index, int stride, int coefficient, float basis
) {
    return add(result, mul(feature(features, index, stride, coefficient), basis));
}

__device__ __forceinline__ V3 rgb(V3 result) {
    return v3(fmaxf(result.x + .5f, 0.f), fmaxf(result.y + .5f, 0.f), fmaxf(result.z + .5f, 0.f));
}

__device__ __forceinline__ V3 eval_background(const float* features, int index, int stride, int degree, V3 direction) {
    const SHBasis b = sh_basis(direction);
    V3 result = v3(0.f, 0.f, 0.f);
    result = add_feature(result, features, index, stride, 0, sh0(b));
    if (degree == 0) return rgb(result);
    result = add_feature(result, features, index, stride, 1, sh1(b));
    result = add_feature(result, features, index, stride, 2, sh2(b));
    result = add_feature(result, features, index, stride, 3, sh3(b));
    if (degree == 1) return rgb(result);
    result = add_feature(result, features, index, stride, 4, sh4(b));
    result = add_feature(result, features, index, stride, 5, sh5(b));
    result = add_feature(result, features, index, stride, 6, sh6(b));
    result = add_feature(result, features, index, stride, 7, sh7(b));
    result = add_feature(result, features, index, stride, 8, sh8(b));
    if (degree == 2) return rgb(result);
    result = add_feature(result, features, index, stride, 9, sh9(b));
    result = add_feature(result, features, index, stride, 10, sh10(b));
    result = add_feature(result, features, index, stride, 11, sh11(b));
    result = add_feature(result, features, index, stride, 12, sh12(b));
    result = add_feature(result, features, index, stride, 13, sh13(b));
    result = add_feature(result, features, index, stride, 14, sh14(b));
    result = add_feature(result, features, index, stride, 15, sh15(b));
    return rgb(result);
}

__device__ __forceinline__ V3 eval_rigid(
    const float* features, int index, int stride, int degree, int degree_t, int max_degree_t,
    float timestamp, V3 direction
) {
    const SHBasis b = sh_basis(direction);
    V3 value = v3(0.f, 0.f, 0.f);
    for (int time_index = 0; time_index <= degree_t; ++time_index) {
        const float time_basis = time_index % 2 == 0
            ? cosf(PI * timestamp * time_index)
            : sinf(PI * timestamp * (time_index + 1));
        value = add(value, mul(feature(features, index, stride, time_index), time_basis));
    }
    V3 result = v3(0.f, 0.f, 0.f);
    result = add(result, mul(value, sh0(b)));
    if (degree == 0) return rgb(result);
    result = add_feature(result, features, index, stride, max_degree_t + 1, sh1(b));
    result = add_feature(result, features, index, stride, max_degree_t + 2, sh2(b));
    result = add_feature(result, features, index, stride, max_degree_t + 3, sh3(b));
    if (degree == 1) return rgb(result);
    result = add_feature(result, features, index, stride, max_degree_t + 4, sh4(b));
    result = add_feature(result, features, index, stride, max_degree_t + 5, sh5(b));
    result = add_feature(result, features, index, stride, max_degree_t + 6, sh6(b));
    result = add_feature(result, features, index, stride, max_degree_t + 7, sh7(b));
    result = add_feature(result, features, index, stride, max_degree_t + 8, sh8(b));
    if (degree == 2) return rgb(result);
    result = add_feature(result, features, index, stride, max_degree_t + 9, sh9(b));
    result = add_feature(result, features, index, stride, max_degree_t + 10, sh10(b));
    result = add_feature(result, features, index, stride, max_degree_t + 11, sh11(b));
    result = add_feature(result, features, index, stride, max_degree_t + 12, sh12(b));
    result = add_feature(result, features, index, stride, max_degree_t + 13, sh13(b));
    result = add_feature(result, features, index, stride, max_degree_t + 14, sh14(b));
    result = add_feature(result, features, index, stride, max_degree_t + 15, sh15(b));
    return rgb(result);
}

__device__ __forceinline__ V3 eval_nonrigid(
    const float* features, int index, int stride, int degree, int degree_t, float mean_t, float timestamp,
    V3 direction
) {
    const SHBasis b = sh_basis(direction);
    V3 result = v3(0.f, 0.f, 0.f);
    result = add_feature(result, features, index, stride, 0, sh0(b));
    if (degree == 0) return rgb(result);
    result = add_feature(result, features, index, stride, 1, sh1(b));
    result = add_feature(result, features, index, stride, 2, sh2(b));
    result = add_feature(result, features, index, stride, 3, sh3(b));
    if (degree == 1) return rgb(result);
    result = add_feature(result, features, index, stride, 4, sh4(b));
    result = add_feature(result, features, index, stride, 5, sh5(b));
    result = add_feature(result, features, index, stride, 6, sh6(b));
    result = add_feature(result, features, index, stride, 7, sh7(b));
    result = add_feature(result, features, index, stride, 8, sh8(b));
    if (degree == 2) return rgb(result);
    result = add_feature(result, features, index, stride, 9, sh9(b));
    result = add_feature(result, features, index, stride, 10, sh10(b));
    result = add_feature(result, features, index, stride, 11, sh11(b));
    result = add_feature(result, features, index, stride, 12, sh12(b));
    result = add_feature(result, features, index, stride, 13, sh13(b));
    result = add_feature(result, features, index, stride, 14, sh14(b));
    result = add_feature(result, features, index, stride, 15, sh15(b));
    const float delta_t = mean_t - timestamp;
    const int coefficient_count = 16;
    for (int time_index = 1; time_index <= degree_t; ++time_index) {
        const float time_basis = cosf(2.f * PI * delta_t * time_index);
        const int offset = time_index * coefficient_count;
        result = add_feature(result, features, index, stride, offset, time_basis * sh0(b));
        result = add_feature(result, features, index, stride, offset + 1, time_basis * sh1(b));
        result = add_feature(result, features, index, stride, offset + 2, time_basis * sh2(b));
        result = add_feature(result, features, index, stride, offset + 3, time_basis * sh3(b));
        result = add_feature(result, features, index, stride, offset + 4, time_basis * sh4(b));
        result = add_feature(result, features, index, stride, offset + 5, time_basis * sh5(b));
        result = add_feature(result, features, index, stride, offset + 6, time_basis * sh6(b));
        result = add_feature(result, features, index, stride, offset + 7, time_basis * sh7(b));
        result = add_feature(result, features, index, stride, offset + 8, time_basis * sh8(b));
        result = add_feature(result, features, index, stride, offset + 9, time_basis * sh9(b));
        result = add_feature(result, features, index, stride, offset + 10, time_basis * sh10(b));
        result = add_feature(result, features, index, stride, offset + 11, time_basis * sh11(b));
        result = add_feature(result, features, index, stride, offset + 12, time_basis * sh12(b));
        result = add_feature(result, features, index, stride, offset + 13, time_basis * sh13(b));
        result = add_feature(result, features, index, stride, offset + 14, time_basis * sh14(b));
        result = add_feature(result, features, index, stride, offset + 15, time_basis * sh15(b));
    }
    return rgb(result);
}

extern "C" __global__ void world_prepare(
    const float* xyz, const float* scales, const float* rotations, const float* opacities,
    const float* ts, const float* scale_ts, const float* rotation_rs, const int* gid_map,
    const float* object_poses, const int* object_active, int n, float timestamp,
    float* world_means, float* sh_means, float* covariances, float* world_opacities
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const int type = gid_map[2 * gid];
    const int object_id = gid_map[2 * gid + 1];
    const float* scale = scales + 3 * gid;
    const float* quaternion = rotations + 4 * gid;
    float covariance[6];
    V3 local_mean = read3(xyz, gid);
    float opacity = opacities[gid];
    if (type == 2) {
        float covariance_t;
        V3 speed;
        conditional_covariance(scale, scale_ts[gid], quaternion, rotation_rs + 4 * gid, covariance, &covariance_t, &speed);
        const float delta_t = timestamp - ts[gid];
        local_mean = add(local_mean, mul(speed, delta_t));
        opacity *= expf(-.5f * delta_t * delta_t / covariance_t);
    } else {
        float rotation[9];
        quaternion_rotation(quaternion, rotation);
        covariance_from_rotation(scale, rotation, covariance);
    }
    V3 world_mean = local_mean;
    if (type != 0) {
        if (!object_active[object_id]) opacity = 0.f;
        const float* pose = object_poses + 16 * object_id;
        world_mean = transform_point(pose, local_mean);
        float world_covariance[6];
        rotate_covariance(pose, covariance, world_covariance);
        for (int index = 0; index < 6; ++index) covariance[index] = world_covariance[index];
    }
    write3(world_means, gid, world_mean);
    write3(sh_means, gid, local_mean);
    float* output_covariance = covariances + 6 * gid;
    for (int index = 0; index < 6; ++index) output_covariance[index] = covariance[index];
    world_opacities[gid] = opacity;
}

extern "C" __global__ void view_prepare(
    const float* world_means, const float* sh_means, const float* world_opacities,
    const float* background_features, const float* rigid_features, const float* nonrigid_features,
    const float* ts, const int* gid_map, const int* object_sources, const float* camera_centers,
    const float* projections, const float* view_matrices, int n,
    int background_feature_stride, int rigid_feature_stride, int nonrigid_feature_stride, float timestamp,
    int background_degree, int rigid_degree, int rigid_degree_t, int rigid_max_degree_t,
    int nonrigid_degree, int nonrigid_degree_t, float* depths, unsigned char* visibility, float* colors
) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const int camera_id = blockIdx.y;
    const int tid = camera_id * n + gid;
    visibility[tid] = 0;
    if (world_opacities[gid] < .004f) return;
    const V3 world_mean = read3(world_means, gid);
    const float* projection = projections + 16 * camera_id;
    const float clip_x = world_mean.x * projection[0] + world_mean.y * projection[4] + world_mean.z * projection[8] + projection[12];
    const float clip_y = world_mean.x * projection[1] + world_mean.y * projection[5] + world_mean.z * projection[9] + projection[13];
    const float clip_z = world_mean.x * projection[2] + world_mean.y * projection[6] + world_mean.z * projection[10] + projection[14];
    const float clip_w = world_mean.x * projection[3] + world_mean.y * projection[7] + world_mean.z * projection[11] + projection[15];
    const float ndc_x = clip_x / clip_w;
    const float ndc_y = clip_y / clip_w;
    const float ndc_z = clip_z / clip_w;
    if (ndc_z <= -1.01f || ndc_z >= 1.01f || ndc_x <= -1.5f || ndc_x >= 1.5f || ndc_y <= -1.5f || ndc_y >= 1.5f) return;
    visibility[tid] = 1;
    const float* view_matrix = view_matrices + 16 * camera_id;
    depths[tid] = world_mean.x * view_matrix[2] + world_mean.y * view_matrix[6] + world_mean.z * view_matrix[10] + view_matrix[11];
    const V3 mean = read3(sh_means, gid);
    const V3 center = read3(camera_centers, camera_id);
    const float dx = mean.x - center.x;
    const float dy = mean.y - center.y;
    const float dz = mean.z - center.z;
    const float inv_norm = rsqrtf(dx * dx + dy * dy + dz * dz);
    const V3 direction = v3(dx * inv_norm, dy * inv_norm, dz * inv_norm);
    const int type = gid_map[2 * gid];
    const int object_id = gid_map[2 * gid + 1];
    const float* features;
    int feature_index;
    int feature_stride;
    if (type == 0) {
        features = background_features;
        feature_index = gid;
        feature_stride = background_feature_stride;
    } else {
        const int* source = object_sources + 2 * object_id;
        feature_index = source[1] + gid - source[0];
        if (type == 1) {
            features = rigid_features;
            feature_stride = rigid_feature_stride;
        } else {
            features = nonrigid_features;
            feature_stride = nonrigid_feature_stride;
        }
    }
    V3 color;
    if (type == 0) {
        color = eval_background(features, feature_index, feature_stride, background_degree, direction);
    } else if (type == 1) {
        color = eval_rigid(features, feature_index, feature_stride, rigid_degree, rigid_degree_t, rigid_max_degree_t, timestamp, direction);
    } else {
        color = eval_nonrigid(features, feature_index, feature_stride, nonrigid_degree, nonrigid_degree_t, ts[gid], timestamp, direction);
    }
    write3(colors, tid, color);
}

extern "C" __global__ void pack_street_vbo(
    float* output, const float* xyz, const float* covariances, const float* colors,
    const float* opacities, const int* gids, const int* draw_counts, int gaussian_count, int draw_count
) {
    const int draw_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (draw_id >= draw_count) return;
    int camera_id = 0;
    int first_after_camera = draw_counts[0];
    while (draw_id >= first_after_camera) first_after_camera += draw_counts[++camera_id];
    const int gid = gids[draw_id];
    float* target = output + 13 * draw_id;
    const float* mean = xyz + 3 * gid;
    const float* covariance = covariances + 6 * gid;
    const float* color = colors + 3 * (camera_id * gaussian_count + gid);
    target[0] = mean[0]; target[1] = mean[1]; target[2] = mean[2];
    target[3] = covariance[0]; target[4] = covariance[1]; target[5] = covariance[2];
    target[6] = covariance[3]; target[7] = covariance[4]; target[8] = covariance[5];
    target[9] = color[0]; target[10] = color[1]; target[11] = color[2];
    target[12] = opacities[gid];
}

extern "C" __global__ void write_draw_commands(
    int* commands, const int* draw_counts, int camera_count
) {
    if (blockIdx.x || threadIdx.x) return;
    int first = 0;
    for (int camera_id = 0; camera_id < camera_count; ++camera_id) {
        int* command = commands + 4 * camera_id;
        command[0] = draw_counts[camera_id];
        command[1] = 1;
        command[2] = first;
        command[3] = 0;
        first += draw_counts[camera_id];
    }
}
'''


def _check_cuda(result, cuda):
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver API failed: {result[0]}")
    return result[1:]


def _check_nvrtc(result, nvrtc):
    if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC failed: {result[0]}")
    return result[1:]


def _kernels():
    global _MODULE, _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    from cuda import cuda, nvrtc

    major, minor = torch.cuda.get_device_capability()
    program, = _check_nvrtc(nvrtc.nvrtcCreateProgram(_SOURCE.encode(), b"street_pipeline.cu", 0, [], []), nvrtc)
    result, = nvrtc.nvrtcCompileProgram(program, 1, [f"--gpu-architecture=compute_{major}{minor}".encode()])
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log_size, = _check_nvrtc(nvrtc.nvrtcGetProgramLogSize(program), nvrtc)
        log = bytearray(log_size)
        _check_nvrtc(nvrtc.nvrtcGetProgramLog(program, log), nvrtc)
        raise RuntimeError(log.decode())
    ptx_size, = _check_nvrtc(nvrtc.nvrtcGetPTXSize(program), nvrtc)
    ptx = bytearray(ptx_size)
    _check_nvrtc(nvrtc.nvrtcGetPTX(program, ptx), nvrtc)
    _MODULE, = _check_cuda(cuda.cuModuleLoadData(bytes(ptx)), cuda)
    _KERNELS = tuple(
        _check_cuda(cuda.cuModuleGetFunction(_MODULE, name), cuda)[0]
        for name in (b"world_prepare", b"view_prepare", b"pack_street_vbo", b"write_draw_commands")
    )
    return _KERNELS


def _launch(kernel, size, values, types, grid_y=1):
    from cuda import cuda

    _check_cuda(
        cuda.cuLaunchKernel(
            kernel,
            (size + 255) // 256,
            grid_y,
            1,
            256,
            1,
            1,
            0,
            torch.cuda.current_stream().cuda_stream,
            (tuple(values), tuple(types)),
            0,
        ),
        cuda,
    )


def pack_vbo(vbo_pointer, means, covariances, colors, opacities, gids, draw_counts):
    _launch(
        _kernels()[2],
        gids.numel(),
        (
            int(vbo_pointer), means.data_ptr(), covariances.data_ptr(), colors.data_ptr(), opacities.data_ptr(),
            gids.data_ptr(), draw_counts.data_ptr(), means.shape[0], gids.numel(),
        ),
        (
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ),
    )


def write_draw_commands(command_pointer, draw_counts):
    _launch(
        _kernels()[3],
        1,
        (int(command_pointer), draw_counts.data_ptr(), draw_counts.numel()),
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int),
    )


def prepare(raw, camera_centers):
    global last_profile

    world_prepare, view_prepare, _, _ = _kernels()
    xyz = raw["xyz"]
    n = xyz.shape[0]
    camera_count = camera_centers.shape[0]
    world_means = torch.empty_like(xyz)
    sh_means = torch.empty_like(xyz)
    covariances = torch.empty((n, 6), dtype=xyz.dtype, device=xyz.device)
    opacities = torch.empty((n,), dtype=xyz.dtype, device=xyz.device)
    colors = torch.empty((camera_count, n, 3), dtype=xyz.dtype, device=xyz.device)
    depths = torch.empty((camera_count, n), dtype=xyz.dtype, device=xyz.device)
    visibility = torch.empty((camera_count, n), dtype=torch.uint8, device=xyz.device)
    pointer = ctypes.c_void_p
    profile = os.environ.get("FAST_GAUSS_BATCH_PROFILE") == "1"
    if profile:
        world_started = torch.cuda.Event(enable_timing=True)
        world_finished = torch.cuda.Event(enable_timing=True)
        view_finished = torch.cuda.Event(enable_timing=True)
        world_started.record()
    _launch(
        world_prepare,
        n,
        (
            raw["xyz"].data_ptr(), raw["scales"].data_ptr(), raw["rotations"].data_ptr(), raw["opacities"].data_ptr(),
            raw["ts"].data_ptr(), raw["scale_ts"].data_ptr(), raw["rotation_rs"].data_ptr(), raw["gid_map"].data_ptr(),
            raw["object_poses"].data_ptr(), raw["object_active"].data_ptr(), n, raw["timestamp"],
            world_means.data_ptr(), sh_means.data_ptr(), covariances.data_ptr(), opacities.data_ptr(),
        ),
        (
            pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
            ctypes.c_int, ctypes.c_float, pointer, pointer, pointer, pointer,
        ),
    )
    if profile:
        world_finished.record()
    _launch(
        view_prepare,
        n,
        (
            world_means.data_ptr(), sh_means.data_ptr(), opacities.data_ptr(),
            raw["background_features"].data_ptr(), raw["rigid_features"].data_ptr(), raw["nonrigid_features"].data_ptr(),
            raw["ts"].data_ptr(), raw["gid_map"].data_ptr(), raw["object_sources"].data_ptr(),
            camera_centers.data_ptr(), raw["projections"].data_ptr(), raw["view_matrices"].data_ptr(), n,
            raw["background_feature_stride"], raw["rigid_feature_stride"], raw["nonrigid_feature_stride"], raw["timestamp"],
            raw["background_degree"], raw["rigid_degree"], raw["rigid_degree_t"], raw["rigid_max_degree_t"],
            raw["nonrigid_degree"], raw["nonrigid_degree_t"], depths.data_ptr(), visibility.data_ptr(), colors.data_ptr(),
        ),
        (
            pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, pointer, pointer, pointer,
        ),
        camera_count,
    )
    if profile:
        view_finished.record()
        view_finished.synchronize()
        last_profile = {
            "street_world_prepare": world_started.elapsed_time(world_finished) / 1000,
            "street_view_prepare": world_finished.elapsed_time(view_finished) / 1000,
        }
    return world_means, covariances, opacities[:, None], colors, depths, visibility
