#version 450 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 inUv[];
layout(location = 1) flat in int inLayer[];

layout(location = 0) out vec3 uv;

void main() {
    for(int i = 0; i < 3; ++i) {
        uv = inUv[i];
        gl_Layer = inLayer[0];
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
