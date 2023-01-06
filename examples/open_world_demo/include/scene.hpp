#pragma once

#include "common.h"
#include <glm/glm.hpp>
#include "VulkanBaseApp.h"

struct SceneData{
    Camera camera;
    struct {
        glm::vec3 position{0};
        float azimuth{0};
        float elevation{0};
    } sun;
    glm::vec3 eyes;
    float time;
    float fieldOfView{90};
    float zNear{1 * meter};
    float zFar{100000 * km};
};

struct GBuffer{
    FramebufferAttachment position;
    FramebufferAttachment normal;
    FramebufferAttachment albedo;
    FramebufferAttachment material;
    FramebufferAttachment edgeDist;
    FramebufferAttachment depth;
};

struct ShadowVolume{
    FramebufferAttachment positionIn;
    FramebufferAttachment positionOut;
    FramebufferAttachment depthStencil;
};