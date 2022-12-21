layout(set = 0, binding = 0) uniform UBO{
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 MVP;

    vec3 sunPosition;
    float maxHeight;

    vec3 wireframeColor;
    int wireframe;

    vec2 numPatches;
    float wireframeWidth;
    int lod;

    float lodMinDepth;
    float lodMaxDepth;
    int minTessLevel;
    int maxTessLevel;

    vec2 viewportSize;
    int shading;
    int tessLevelColor;

    float lodTargetTriangleWidth;
    int lodStrategy;
};
