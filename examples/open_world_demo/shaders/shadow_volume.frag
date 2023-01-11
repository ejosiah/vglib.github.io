#version 460

layout(input_attachment_index=0, set=0, binding=1) uniform subpassInput shaodwInDepth;

layout(location = 0) out vec4 shadowInOut;

void main(){
    shadowInOut.x = subpassLoad(shaodwInDepth).r;
    shadowInOut.y = gl_FragCoord.z;
}