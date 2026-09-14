#version 330
#extension GL_ARB_viewport_array : require

uniform vec2 uBasisViewport[32];
uniform float discardAlpha = 0.0001;
uniform float sqrt8 = sqrt(8);

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec2 basisVector0[];
in vec2 basisVector1[];
in vec4 gColor[];
in float gDepth[];
flat in int gLayer[];

out vec2 vPosition;
flat out vec4 vColor;
flat out float vDepth;

void emitCorner(vec2 corner, vec3 ndcCenter, vec2 basisViewport) {
    vec2 offset = (corner.x * basisVector0[0] + corner.y * basisVector1[0]) * basisViewport * 2.0;
    gl_Position = vec4(ndcCenter.xy + offset, ndcCenter.z, 1.0);
    vPosition = corner * sqrt8;
    vColor = gColor[0];
    vDepth = gDepth[0];
    EmitVertex();
}

void main() {
    if (gColor[0].a < discardAlpha) return;
    int layer = gLayer[0];
    gl_Layer = layer;
    vec3 ndcCenter = gl_in[0].gl_Position.xyz;
    vec2 basisViewport = uBasisViewport[layer];
    emitCorner(vec2(-1.0, -1.0), ndcCenter, basisViewport);
    emitCorner(vec2(-1.0,  1.0), ndcCenter, basisViewport);
    emitCorner(vec2( 1.0, -1.0), ndcCenter, basisViewport);
    emitCorner(vec2( 1.0,  1.0), ndcCenter, basisViewport);
    EndPrimitive();
}
