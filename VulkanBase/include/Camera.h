#pragma once

#include "camera_base.h"
#include "OrbitingCamera.h"
#include "FirstPersonCamera.h"
#include "FlightCamera.h"

enum class CameraMode{
    FIRST_PERSON,
    SPECTATOR,
    FLIGHT,
    ORBIT,
    NONE
};

struct CameraSettings : BaseCameraSettings{
    FirstPersonSpectatorCameraSettings firstPerson;
    FlightCameraSettings flight;
    OrbitingCameraSettings orbit;
    CameraMode mode = CameraMode::NONE;
};

class CameraController final : public AbstractCamera {
public:
    CameraController( InputManager& inputManager, const CameraSettings& settings);

    ~CameraController() override = default;

    void update(float time) final;

    void processInput() final;

    void setMode(CameraMode mode);

    void lookAt(const glm::vec3 &eye, const glm::vec3 &target, const glm::vec3 &up) final;

    void perspective(float fovx, float aspect, float znear, float zfar) final;

    void perspective(float aspect) final;

    void rotateSmoothly(float headingDegrees, float pitchDegrees, float rollDegrees) final;

    void rotate(float headingDegrees, float pitchDegrees, float rollDegrees) final;

    void move(float dx, float dy, float dz) final;

    void move(const glm::vec3 &direction, const glm::vec3 &amount) final;

    void position(const glm::vec3& pos) final;

    void updatePosition(const glm::vec3 &direction, float elapsedTimeSec) final;

    void undoRoll() final;

    void zoom(float zoom, float minZoom, float maxZoom) final;

    void onResize(int width, int height) final;

    void setModel(const glm::mat4& model) final;

    void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) const final;

    void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const glm::mat4& model, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) final;

    const glm::vec3 &position() const final;

    const glm::vec3 &velocity() const final;

    const glm::vec3 &acceleration() const final;

    float near() const final;

    float far() const final;

    std::string mode() const;

    [[nodiscard]]
    const Camera& cam() const final;

    bool isInFirstPersonMode() const;

    bool isInSpectatorMode() const;

    bool isInFlightMode() const;

    bool isInObitMode() const;

    const glm::quat &getOrientation() const final;

    void newFrame() override;

    bool moved() const override;

    void fieldOfView(float value) override;

    const Camera &previousCamera() const override;

    void jitter(float jx, float jy) override;

    void extract(Frustum &frustum) const override;

    void extractAABB(glm::vec3 &bMin, glm::vec3 &bMax) const override;

private:
    CameraMode currentMode;
    mutable std::map<CameraMode, std::unique_ptr<BaseCameraController>> cameras;
    Action& firstPerson;
    Action& spectator;
    Action& flight;
    Action& orbit;

};