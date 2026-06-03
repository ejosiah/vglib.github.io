#pragma once

#include "camera_base.h"
#include "OrbitingCamera.h"
#include "FirstPersonCamera.h"
#include "FlightCamera.h"

enum class CameraMode {
    FIRST_PERSON,
    SPECTATOR,
    FLIGHT,
    ORBIT,
    NONE
};

template<typename Scalar>
struct CameraSettingsT : BaseCameraSettingsT<Scalar> {
    FirstPersonSpectatorCameraSettingsT<Scalar> firstPerson;
    FlightCameraSettingsT<Scalar> flight;
    OrbitingCameraSettingsT<Scalar> orbit;
    CameraMode mode = CameraMode::NONE;
};

template<typename Scalar>
class CameraControllerT final : public AbstractCameraT<Scalar> {
public:
    using Vec3 = glm::vec<3, Scalar, glm::defaultp>;
    using Mat4 = glm::mat<4, 4, Scalar, glm::defaultp>;
    using Quat = glm::qua<Scalar, glm::defaultp>;
    using Camera = CameraT<Scalar>;
    using Frustum = FrustumT<Scalar>;
    using Settings = CameraSettingsT<Scalar>;

    CameraControllerT(InputManager& inputManager, const Settings& settings);

    ~CameraControllerT() override = default;

    void update(float time) final;

    void processInput() final;

    void setMode(CameraMode mode);

    void lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) final;

    void perspective(Scalar fovx, Scalar aspect, Scalar znear, Scalar zfar) final;

    void perspective(Scalar aspect) final;

    void rotateSmoothly(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) final;

    void rotate(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) final;

    void move(Scalar dx, Scalar dy, Scalar dz) final;

    void move(const Vec3& direction, const Vec3& amount) final;

    void position(const Vec3& pos) final;

    void updatePosition(const Vec3& direction, Scalar elapsedTimeSec) final;

    void undoRoll() final;

    void zoom(Scalar zoom, Scalar minZoom, Scalar maxZoom) final;

    void onResize(int width, int height) final;

    void setModel(const Mat4& model) final;

    void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) const final;

    void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const Mat4& model, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) final;

    const Vec3& position() const final;

    const Vec3& velocity() const final;

    const Vec3& acceleration() const final;

    Vec3 viewDirection() const;

    Scalar near() const final;

    Scalar far() const final;

    void near(float value);

    void far(float value);

    std::string modeToString() const;

    CameraMode mode() const;

    [[nodiscard]]
    const Camera& cam() const final;

    bool isInFirstPersonMode() const;

    bool isInSpectatorMode() const;

    bool isInFlightMode() const;

    bool isInObitMode() const;

    const Quat& getOrientation() const final;

    void newFrame() override;

    bool moved() const override;

    void fieldOfView(Scalar value) override;

    Scalar fieldOfView() const;

    const Camera& previousCamera() const override;

    void jitter(Scalar jx, Scalar jy) override;

    void extract(Frustum& frustum) const override;

    void extractAABB(Vec3& bMin, Vec3& bMax) const override;

    Camera cameraMatrix() const;

    Scalar aspectRatio();

private:
    void resetPerspective();

    CameraMode currentMode;
    mutable std::map<CameraMode, std::unique_ptr<BaseCameraControllerT<Scalar>>> cameras;
    Action& firstPerson;
    Action& spectator;
    Action& flight;
    Action& orbit;
};

using CameraSettings = CameraSettingsT<float>;
using DoubleCameraSettings = CameraSettingsT<double>;

using CameraController = CameraControllerT<float>;
using DoubleCameraController = CameraControllerT<double>;
