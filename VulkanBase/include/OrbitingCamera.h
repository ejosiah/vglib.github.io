#pragma once

#include "camera_base.h"
#include <limits>

static constexpr float DEFAULT_ORBIT_MIN_ZOOM = DEFAULT_ZOOM_MIN;
static constexpr float DEFAULT_ORBIT_MAX_ZOOM = DEFAULT_ZOOM_MAX;
static constexpr float DEFAULT_SPEED_ORBIT_ROLL = 100.0f;
static constexpr float DEFAULT_ORBIT_OFFSET_DISTANCE = DEFAULT_ORBIT_MIN_ZOOM + (DEFAULT_ORBIT_MAX_ZOOM - DEFAULT_ORBIT_MIN_ZOOM) * 0.25f;

template<typename Scalar>
struct OrbitingCameraSettingsT : public BaseCameraSettingsT<Scalar> {
    using Vec3 = glm::vec<3, Scalar, glm::defaultp>;

    Scalar offsetDistance = Scalar(DEFAULT_ORBIT_OFFSET_DISTANCE);
    Scalar orbitRollSpeed = Scalar(DEFAULT_SPEED_ORBIT_ROLL);
    Scalar orbitMinZoom = Scalar(DEFAULT_ORBIT_MIN_ZOOM);
    Scalar orbitMaxZoom = Scalar(DEFAULT_ORBIT_MAX_ZOOM);
    Scalar modelHeight = Scalar(1);
    bool preferTargetYAxisOrbiting = true;
    Vec3 target{std::numeric_limits<Scalar>::quiet_NaN()};
    struct {
        Vec3 min{std::numeric_limits<Scalar>::max()};
        Vec3 max{std::numeric_limits<Scalar>::lowest()};
    } model;
};

// FIXME change to focus on a target and not a model in the scene
template<typename Scalar>
class OrbitingCameraControllerT : public BaseCameraControllerT<Scalar> {
public:
    using Base = BaseCameraControllerT<Scalar>;
    using Vec3 = typename Base::Vec3;
    using Mat4 = typename Base::Mat4;
    using Quat = typename Base::Quat;
    using Settings = OrbitingCameraSettingsT<Scalar>;

    OrbitingCameraControllerT(InputManager& inputManager, const Settings& settings = {});

    void update(float elapsedTime) override;

    void updateModel(const Vec3& position, const Quat& orientation = {Scalar(1), Scalar(0), Scalar(0), Scalar(0)});

    void updateModel(const Vec3& bMin, const Vec3& bMax);

    void move(Scalar dx, Scalar dy, Scalar dz) override;

    void move(const Vec3& direction, const Vec3& amount) override;

    void rotate(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) override;

    void undoRoll() override;

    void zoom(Scalar zoom, Scalar minZoom, Scalar maxZoom) override;

    void updateViewMatrix() override;

    void onPositionChanged() final;

    void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) const override;

    void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const Mat4& model, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) override;

    [[nodiscard]]
    Mat4 getModel() const;

private:
    Scalar offsetDistance;
    Scalar orbitRollSpeed;
    bool preferTargetYAxisOrbiting;

    mutable struct {
        Vec3 position;
        Quat orientation;
    } model;
};

using OrbitingCameraSettings = OrbitingCameraSettingsT<float>;
using DoubleOrbitingCameraSettings = OrbitingCameraSettingsT<double>;

using OrbitingCameraController = OrbitingCameraControllerT<float>;
using DoubleOrbitingCameraController = OrbitingCameraControllerT<double>;
