#version 330
#extension GL_ARB_shader_draw_parameters : require

#define MAX_BATCH_CAMERAS 32

uniform int cameraCount;
uniform mat4 uP[MAX_BATCH_CAMERAS];
uniform mat4 uVM[MAX_BATCH_CAMERAS];
uniform vec2 uFocal[MAX_BATCH_CAMERAS];
uniform vec2 uPrincipal[MAX_BATCH_CAMERAS];
uniform vec2 uBasisViewport[MAX_BATCH_CAMERAS];

uniform float discardAlpha = 0.0001;
uniform float maxScreenSpaceSplatSize = 2048.0;
uniform float sqrt8 = sqrt(8);

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aCov0_3;
layout(location = 2) in vec3 aCov3_6;
layout(location = 3) in vec4 aColor;

out vec2 basisVector0;
out vec2 basisVector1;
out vec4 gColor;
out float gDepth;
flat out int gLayer;

void main() {
    int cameraId = gl_DrawIDARB;
    if (cameraId >= cameraCount || aColor.a < discardAlpha) {
        gColor.a = 0.0;
        return;
    }

    mat4 P = uP[cameraId];
    mat4 VM = uVM[cameraId];
    vec2 focal = uFocal[cameraId];
    vec2 principal = uPrincipal[cameraId];
    vec2 basisViewport = uBasisViewport[cameraId];

    vec4 viewCenter = VM * vec4(aPos, 1.0);
    vec4 clipCenter = P * vec4(aPos, 1.0);
    clipCenter = clipCenter / clipCenter.w;

    mat3 Vrk = mat3(
        aCov0_3[0], aCov0_3[1], aCov0_3[2],
        aCov0_3[1], aCov3_6[0], aCov3_6[1],
        aCov0_3[2], aCov3_6[1], aCov3_6[2]
    );

    float width = 1.0 / basisViewport[0];
    float height = 1.0 / basisViewport[1];
    float fx = focal[0];
    float fy = focal[1];
    float cx = principal[0];
    float cy = principal[1];
    float x = viewCenter.x;
    float y = viewCenter.y;
    float z = viewCenter.z;

    float tanFovx = 0.5 * width / fx;
    float tanFovy = 0.5 * height / fy;
    float limXPos = (width - cx) / fx + 0.3 * tanFovx;
    float limXNeg = cx / fx + 0.3 * tanFovx;
    float limYPos = (height - cy) / fy + 0.3 * tanFovy;
    float limYNeg = cy / fy + 0.3 * tanFovy;

    float rz = 1.0 / z;
    float rz2 = rz * rz;
    float tx = z * min(limXPos, max(-limXNeg, x * rz));
    float ty = z * min(limYPos, max(-limYNeg, y * rz));
    mat3 J = mat3(fx * rz, 0., -fx * tx * rz2, 0., fy * rz, -fy * ty * rz2, 0., 0., 0.);
    mat3 T = transpose(mat3(VM)) * J;
    mat3 cov2Dm = transpose(T) * Vrk * T;
    cov2Dm[0][0] += 0.3;
    cov2Dm[1][1] += 0.3;

    float a = cov2Dm[0][0];
    float b = cov2Dm[0][1];
    float d = cov2Dm[1][1];
    float D = a * d - b * b;
    if (D <= 0.0 || a <= 0.0 || d <= 0.0) {
        gColor.a = 0.0;
        return;
    }

    float traceOver2 = 0.5 * (a + d);
    float term2 = sqrt(max(0.1, traceOver2 * traceOver2 - D));
    float eigenValue0 = traceOver2 + term2;
    float eigenValue1 = traceOver2 - term2;
    if (eigenValue0 <= 0.01 || eigenValue1 <= 0.01 || eigenValue0 < eigenValue1 || eigenValue0 / eigenValue1 > 10000.0) {
        gColor.a = 0.0;
        return;
    }

    vec2 eigenVector0 = normalize(vec2(b, eigenValue0 - a));
    vec2 eigenVector1 = vec2(eigenVector0.y, -eigenVector0.x);
    basisVector0 = eigenVector0 * min(sqrt8 * sqrt(eigenValue0), maxScreenSpaceSplatSize);
    basisVector1 = eigenVector1 * min(sqrt8 * sqrt(eigenValue1), maxScreenSpaceSplatSize);
    gl_Position = clipCenter;
    gColor = aColor;
    gDepth = clipCenter.z;
    gLayer = cameraId;
}
