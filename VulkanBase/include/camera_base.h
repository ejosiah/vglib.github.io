#pragma once

#include "common.h"
#include "InputManager.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "VulkanDevice.h"
#include "AbstractCamera.hpp"

static constexpr float HALF_PI = glm::half_pi<float>();
static constexpr float PI = glm::pi<float>();
static constexpr float TWO_PI = glm::two_pi<float>();

static constexpr float DEFAULT_ROTATION_SPEED = 0.3f;
static constexpr float DEFAULT_FOVX = 90.0f;
static constexpr float DEFAULT_ZNEAR = 0.1f;
static constexpr float DEFAULT_ZFAR = 1000.0f;
static constexpr float DEFAULT_ZOOM_MAX = 5.0f;
static constexpr float DEFAULT_ZOOM_MIN = 1.5f;

static constexpr glm::vec3 DEFAULT_ACCELERATION(4.0f, 4.0f, 4.0f);
static constexpr glm::vec3 DEFAULT_VELOCITY(1.0f);

template<typename Scalar>
inline constexpr glm::vec<3, Scalar, glm::defaultp> WORLD_XAXIS_T(Scalar(1), Scalar(0), Scalar(0));

template<typename Scalar>
inline constexpr glm::vec<3, Scalar, glm::defaultp> WORLD_YAXIS_T(Scalar(0), Scalar(1), Scalar(0));

template<typename Scalar>
inline constexpr glm::vec<3, Scalar, glm::defaultp> WORLD_ZAXIS_T(Scalar(0), Scalar(0), Scalar(1));

constexpr glm::vec3 WORLD_XAXIS = WORLD_XAXIS_T<float>;
constexpr glm::vec3 WORLD_YAXIS = WORLD_YAXIS_T<float>;
constexpr glm::vec3 WORLD_ZAXIS = WORLD_ZAXIS_T<float>;

template<typename Scalar>
struct BaseCameraSettingsT {
    using Vec3 = glm::vec<3, Scalar, glm::defaultp>;

    Vec3 acceleration = Vec3(Scalar(4), Scalar(4), Scalar(4));
    Vec3 velocity = Vec3(Scalar(1));
    Scalar rotationSpeed = Scalar(DEFAULT_ROTATION_SPEED);
    Scalar fieldOfView = Scalar(DEFAULT_FOVX);
    Scalar aspectRatio = Scalar(1);
    Scalar zNear = Scalar(DEFAULT_ZNEAR);
    Scalar zFar = Scalar(DEFAULT_ZFAR);
    Scalar minZoom = Scalar(DEFAULT_ZOOM_MIN);
    Scalar maxZoom = Scalar(DEFAULT_ZOOM_MAX);
    Scalar floorOffset = Scalar(0.5);
    bool handleZoom = false;
    bool horizontalFov = false;
};

template<typename Scalar>
struct BaseCameraControllerT : public AbstractCameraT<Scalar> {
public:
    using Vec3 = glm::vec<3, Scalar, glm::defaultp>;
    using Vec4 = glm::vec<4, Scalar, glm::defaultp>;
    using Mat4 = glm::mat<4, 4, Scalar, glm::defaultp>;
    using Quat = glm::qua<Scalar, glm::defaultp>;
    using Camera = CameraT<Scalar>;
    using Frustum = FrustumT<Scalar>;
    using Settings = BaseCameraSettingsT<Scalar>;

    BaseCameraControllerT(InputManager& inputManager, const Settings& settings = {});

    ~BaseCameraControllerT() override = default;

    void processInput() override;

    void lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) final;

    void perspective(Scalar fovx, Scalar aspect, Scalar znear, Scalar zfar) final;

    void perspective(Scalar aspect) final;

    void rotateSmoothly(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) override;

    void move(Scalar dx, Scalar dy, Scalar dz) override;

    void move(const Vec3& direction, const Vec3& amount) override;

    void position(const Vec3& pos) final;

    [[nodiscard]]
    const Vec3& position() const final;

    virtual void onPositionChanged();

    [[nodiscard]]
    const Vec3& velocity() const final;

    [[nodiscard]]
    const Vec3& acceleration() const final;

    void updatePosition(const Vec3& direction, Scalar elapsedTimeSec) override;

    void undoRoll() override;

    void zoom(Scalar zoom, Scalar minZoom, Scalar maxZoom) override;

    void onResize(int width, int height) override;

    void setModel(const Mat4& model) override;

    void setTargetYAxis(const Vec3& axis);

    const Vec3& getYAxis();

    Scalar near() const override;

    Scalar far() const override;

    void fieldOfView(Scalar value) override;

    void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) const override;

    void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const Mat4& model, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) override;

    [[nodiscard]]
    const Camera& cam() const final;

    const Camera& previousCamera() const override;

    [[nodiscard]]
    const Quat& getOrientation() const final;

public:
    virtual void updateViewMatrix();

    virtual void processMovementInput();

    virtual void processZoomInput();

    virtual void updateVelocity(const Vec3& direction, Scalar elapsedTimeSec);

    void newFrame() override;

    bool moved() const override;

    void jitter(Scalar jx, Scalar jy) final;

    void extract(Frustum& frustum) const final;

    void extractAABB(Vec3& bMin, Vec3& bMax) const override;

    Scalar fov;
    Scalar aspectRatio;
    Scalar znear;
    Scalar zfar;
    Scalar minZoom;
    Scalar maxZoom;
    Scalar zoomDelta = Scalar(0.1);
    Scalar rotationSpeed;
    Scalar accumPitchDegrees;
    Scalar floorOffset;
    bool handleZoom;
    bool horizontalFov;
    Vec3 eyes;
    Vec3 target;
    Vec3 targetYAxis;
    Vec3 xAxis;
    Vec3 yAxis;
    Vec3 zAxis;
    Vec3 viewDir;
    Vec3 _acceleration;
    Vec3 currentVelocity;
    Vec3 _velocity;
    Quat orientation;
    Vec3 direction;
    mutable Camera camera;
    mutable Camera _previousCamera;
    const Mouse& mouse;
    mutable std::array<Vec4, 8> corners{};

    Scalar zoomAmount = Scalar(0);

    struct {
        Action* forward;
        Action* back;
        Action* left;
        Action* right;
        Action* up;
        Action* down;
    } _move{};

    Action& zoomIn;
    Action& zoomOut;

    bool _moved;
};

using BaseCameraSettings = BaseCameraSettingsT<float>;
using DoubleBaseCameraSettings = BaseCameraSettingsT<double>;

using BaseCameraController = BaseCameraControllerT<float>;
using DoubleBaseCameraController = BaseCameraControllerT<double>;
