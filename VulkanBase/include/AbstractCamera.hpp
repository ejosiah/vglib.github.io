#pragma once

#include "VulkanRAII.h"

#include <glm/glm.hpp>
#include <array>

struct Camera{
    glm::mat4 model = glm::mat4(1);
    glm::mat4 view = glm::mat4(1);
    glm::mat4 proj = glm::mat4(1);

    static constexpr VkPushConstantRange pushConstant(VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) {
        return {stageFlags, 0, sizeof(Camera)};
    }

};

enum PlaneType : int { LEFT_PLANE = 0, RIGHT_PLANE, BOTTOM_PLANE, TOP_PLANE, NEAR_PLANE, FAR_PLANE};

using ClipPlane = glm::vec4;

struct Frustum {
    std::array<ClipPlane, 6> cp;

    bool test(const glm::vec3& point) const;

    bool test(const glm::vec3& boxMin, const glm::vec3& boxMax) const;

    bool test(const glm::vec3& boxCenter, float scale);

    static void extractFrustum(Frustum& frustum, const glm::mat4 M);
};

class AbstractCamera {
public:
    virtual ~AbstractCamera() = default;

    virtual void update(float time) = 0;

    virtual void processInput() = 0;

    virtual void lookAt(const glm::vec3 &eye, const glm::vec3 &target, const glm::vec3 &up) = 0;

    virtual void perspective(float fovx, float aspect, float znear, float zfar) = 0;

    virtual void perspective(float aspect) = 0;

    virtual void rotateSmoothly(float headingDegrees, float pitchDegrees, float rollDegrees) = 0;

    virtual void rotate(float headingDegrees, float pitchDegrees, float rollDegrees) = 0;

    virtual void move(float dx, float dy, float dz) = 0;

    virtual void move(const glm::vec3 &direction, const glm::vec3 &amount) = 0;

    virtual void position(const glm::vec3& pos) = 0;

    [[nodiscard]]
    virtual const glm::vec3& position() const = 0;

    [[nodiscard]]
    virtual const glm::vec3& velocity() const = 0;

    [[nodiscard]]
    virtual const glm::vec3& acceleration() const = 0;

    virtual float near() = 0;

    virtual float far() = 0;

    virtual void fieldOfView(float value) = 0;

    virtual void updatePosition(const glm::vec3 &direction, float elapsedTimeSec) = 0;

    virtual void undoRoll() = 0;

    virtual void zoom(float zoom, float minZoom, float maxZoom) = 0;

    virtual void onResize(int width, int height) = 0;

    virtual void setModel(const glm::mat4& model) = 0;

    virtual void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) const = 0;

    virtual void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const glm::mat4& model, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) = 0;

    [[nodiscard]]
    virtual const glm::quat& getOrientation() const = 0;

    [[nodiscard]]
    virtual const Camera& cam() const = 0;

    virtual const Camera& previousCamera() const = 0;

    virtual void newFrame() = 0;

    [[nodiscard]]
    virtual bool moved() const = 0;

    virtual void jitter(float jx, float jy) = 0;

    virtual void extract(Frustum& frustum) const = 0;

    virtual void extractAABB(glm::vec3& bMin, glm::vec3& bMax) const = 0;

};