#include "camera_base.h"
#include "AbstractCamera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>

namespace {
    template<typename Scalar>
    bool close_enough(Scalar x, Scalar y, Scalar epsilon = Scalar(1E-3)) {
        return std::abs(x - y) <= epsilon * (std::abs(x) + std::abs(y) + Scalar(1));
    }

    template<typename Scalar>
    glm::mat<4, 4, Scalar, glm::defaultp> perspective_vfov_vk(Scalar fovy, Scalar aspect, Scalar zNear, Scalar zFar) {
        auto result = glm::perspectiveRH_ZO(fovy, aspect, zNear, zFar);
        result[1][1] *= Scalar(-1);
        return result;
    }

    template<typename Scalar>
    glm::mat<4, 4, Scalar, glm::defaultp> perspective_hfov_vk(Scalar fovx, Scalar aspect, Scalar zNear, Scalar zFar) {
        const auto fovy = Scalar(2) * std::atan(std::tan(fovx * Scalar(0.5)) / aspect);
        return perspective_vfov_vk(fovy, aspect, zNear, zFar);
    }

    template<typename Scalar>
    glm::mat<4, 4, Scalar, glm::defaultp> perspective_matrix(Scalar fov, Scalar aspect, Scalar zNear, Scalar zFar, bool horizontalFov) {
        return horizontalFov ? perspective_hfov_vk(fov, aspect, zNear, zFar)
                             : perspective_vfov_vk(fov, aspect, zNear, zFar);
    }
}

template<typename Scalar>
BaseCameraControllerT<Scalar>::BaseCameraControllerT(InputManager& inputManager, const Settings& settings)
    : fov(settings.fieldOfView)
    , aspectRatio(settings.aspectRatio)
    , znear(settings.zNear)
    , zfar(settings.zFar)
    , minZoom(settings.minZoom)
    , maxZoom(settings.maxZoom)
    , rotationSpeed(settings.rotationSpeed)
    , accumPitchDegrees(Scalar(0))
    , floorOffset(settings.floorOffset)
    , handleZoom(settings.handleZoom)
    , horizontalFov(settings.horizontalFov)
    , eyes(Scalar(0))
    , target(Scalar(0))
    , targetYAxis(Scalar(0), Scalar(1), Scalar(0))
    , xAxis(Scalar(1), Scalar(0), Scalar(0))
    , yAxis(Scalar(0), Scalar(1), Scalar(0))
    , zAxis(Scalar(0), Scalar(0), Scalar(1))
    , viewDir(Scalar(0), Scalar(0), Scalar(-1))
    , _acceleration(settings.acceleration)
    , currentVelocity(Scalar(0))
    , _velocity(settings.velocity)
    , orientation(Scalar(1), Scalar(0), Scalar(0), Scalar(0))
    , direction(Scalar(0))
    , camera()
    , mouse(inputManager.getMouse())
    , zoomIn(inputManager.mapToMouse(MouseEvent::MoveCode::WHEEL_UP))
    , zoomOut(inputManager.mapToMouse(MouseEvent::MoveCode::WHEEL_DOWN))
{
    _move.forward = &inputManager.mapToKey(Key::W, "forward", Action::Behavior::DETECT_INITIAL_PRESS_ONLY);
    _move.back = &inputManager.mapToKey(Key::S, "backward", Action::Behavior::DETECT_INITIAL_PRESS_ONLY);
    _move.left = &inputManager.mapToKey(Key::A, "left", Action::Behavior::DETECT_INITIAL_PRESS_ONLY);
    _move.right = &inputManager.mapToKey(Key::D, "right", Action::Behavior::DETECT_INITIAL_PRESS_ONLY);
    _move.up = &inputManager.mapToKey(Key::E, "up", Action::Behavior::DETECT_INITIAL_PRESS_ONLY);
    _move.down = &inputManager.mapToKey(Key::Q, "down", Action::Behavior::DETECT_INITIAL_PRESS_ONLY);
    position({Scalar(0), floorOffset, Scalar(0)});
    perspective(fov, aspectRatio, znear, zfar);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::processInput() {
    processMovementInput();
    processZoomInput();
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::processMovementInput() {
    direction = Vec3(Scalar(0));
    auto vel = currentVelocity;
    if (_move.forward->isPressed()) {
        currentVelocity.x = vel.x;
        currentVelocity.y = vel.y;
        currentVelocity.z = Scalar(0);
    } else if (_move.forward->isHeld()) {
        direction.z += Scalar(1);
    }

    if (_move.back->isPressed()) {
        currentVelocity.x = vel.x;
        currentVelocity.y = vel.y;
        currentVelocity.z = Scalar(0);
    } else if (_move.back->isHeld()) {
        direction.z -= Scalar(1);
    }

    if (_move.right->isPressed()) {
        currentVelocity.x = Scalar(0);
        currentVelocity.y = vel.y;
        currentVelocity.z = vel.z;
    } else if (_move.right->isHeld()) {
        direction.x += Scalar(1);
    }

    if (_move.left->isPressed()) {
        currentVelocity.x = Scalar(0);
        currentVelocity.y = vel.y;
        currentVelocity.z = vel.z;
    } else if (_move.left->isHeld()) {
        direction.x -= Scalar(1);
    }

    if (_move.up->isPressed()) {
        currentVelocity.x = vel.x;
        currentVelocity.y = Scalar(0);
        currentVelocity.z = vel.z;
    } else if (_move.up->isHeld()) {
        direction.y += Scalar(1);
    }

    if (_move.down->isPressed()) {
        currentVelocity.x = vel.x;
        currentVelocity.y = Scalar(0);
        currentVelocity.z = vel.z;
    } else if (_move.down->isHeld()) {
        direction.y -= Scalar(1);
    }
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::processZoomInput() {
    zoomAmount = Scalar(0);
    if (zoomIn.isPressed()) {
        zoomAmount = -zoomDelta;
    } else if (zoomOut.isPressed()) {
        zoomAmount = zoomDelta;
    }
    if (handleZoom && zoomAmount != Scalar(0)) {
        zoom(zoomAmount, minZoom, maxZoom);
    }
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    this->eyes = eye;
    this->target = target;

    auto& view = camera.view;
    view = glm::lookAt(eye, target, up);
    accumPitchDegrees = glm::degrees(std::asin(view[1][2]));

    xAxis = Vec3(row(view, 0));
    yAxis = Vec3(row(view, 1));
    zAxis = Vec3(row(view, 2));

    viewDir = -zAxis;

    accumPitchDegrees = glm::degrees(std::asin(view[1][2]));

    orientation = Quat(view);
    updateViewMatrix();
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::perspective(Scalar aspect) {
    perspective(fov, aspect, znear, zfar);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::perspective(Scalar fov, Scalar aspect, Scalar znear, Scalar zfar) {
    camera.proj = perspective_matrix(glm::radians(fov), aspect, znear, zfar, horizontalFov);
    this->fov = fov;
    aspectRatio = aspect;
    this->znear = znear;
    this->zfar = zfar;
    _moved = true;
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::rotateSmoothly(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) {
    headingDegrees *= rotationSpeed;
    pitchDegrees *= rotationSpeed;
    rollDegrees *= rotationSpeed;

    this->rotate(headingDegrees, pitchDegrees, rollDegrees);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::undoRoll() {
    lookAt(eyes, eyes + viewDir, WORLD_YAXIS_T<Scalar>);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::zoom(Scalar zoom, Scalar minZoom, Scalar maxZoom) {
    zoom = std::min(std::max(zoom, minZoom), maxZoom);
    perspective(zoom, aspectRatio, znear, zfar);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::move(Scalar dx, Scalar dy, Scalar dz) {
    if (dx == Scalar(0) && dy == Scalar(0) && dz == Scalar(0)) return;

    Vec3 eyes = this->eyes;
    Vec3 forwards = viewDir;

    eyes += xAxis * dx;
    eyes += WORLD_YAXIS_T<Scalar> * dy;
    eyes += forwards * dz;

    position(eyes);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::move(const Vec3& direction, const Vec3& amount) {
    eyes.x += direction.x * amount.x;
    eyes.y += direction.y * amount.y;
    eyes.z += direction.z * amount.z;

    updateViewMatrix();
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::position(const Vec3& pos) {
    eyes = pos;
    onPositionChanged();
    updateViewMatrix();
}

template<typename Scalar>
const typename BaseCameraControllerT<Scalar>::Vec3& BaseCameraControllerT<Scalar>::position() const {
    return eyes;
}

template<typename Scalar>
const typename BaseCameraControllerT<Scalar>::Vec3& BaseCameraControllerT<Scalar>::velocity() const {
    return currentVelocity;
}

template<typename Scalar>
const typename BaseCameraControllerT<Scalar>::Vec3& BaseCameraControllerT<Scalar>::acceleration() const {
    return _acceleration;
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::updatePosition(const Vec3& direction, Scalar elapsedTimeSec) {
    using namespace glm;
    if (dot(currentVelocity, currentVelocity) != Scalar(0)) {
        Vec3 displacement = (currentVelocity * elapsedTimeSec) +
                            (Scalar(0.5) * _acceleration * elapsedTimeSec * elapsedTimeSec);

        if (direction.x == Scalar(0) && close_enough(currentVelocity.x, Scalar(0)))
            displacement.x = Scalar(0);

        if (direction.y == Scalar(0) && close_enough(currentVelocity.y, Scalar(0)))
            displacement.y = Scalar(0);

        if (direction.z == Scalar(0) && close_enough(currentVelocity.z, Scalar(0)))
            displacement.z = Scalar(0);

        move(displacement.x, displacement.y, displacement.z);
    }

    updateVelocity(direction, elapsedTimeSec);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::updateViewMatrix() {
    auto& view = camera.view;
    view = glm::mat4_cast(orientation);

    xAxis = Vec3(glm::row(view, 0));
    yAxis = Vec3(glm::row(view, 1));
    zAxis = Vec3(glm::row(view, 2));
    viewDir = -zAxis;

    view[3][0] = -dot(xAxis, eyes);
    view[3][1] = -dot(yAxis, eyes);
    view[3][2] = -dot(zAxis, eyes);
    _moved = true;
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::updateVelocity(const Vec3& direction, Scalar elapsedTimeSec) {
    if (direction.x != Scalar(0)) {
        currentVelocity.x += direction.x * _acceleration.x * elapsedTimeSec;

        if (currentVelocity.x > _velocity.x)
            currentVelocity.x = _velocity.x;
        else if (currentVelocity.x < -_velocity.x)
            currentVelocity.x = -_velocity.x;
    } else {
        if (currentVelocity.x > Scalar(0)) {
            if ((currentVelocity.x -= _acceleration.x * elapsedTimeSec) < Scalar(0))
                currentVelocity.x = Scalar(0);
        } else {
            if ((currentVelocity.x += _acceleration.x * elapsedTimeSec) > Scalar(0))
                currentVelocity.x = Scalar(0);
        }
    }

    if (direction.y != Scalar(0)) {
        currentVelocity.y += direction.y * _acceleration.y * elapsedTimeSec;

        if (currentVelocity.y > _velocity.y)
            currentVelocity.y = _velocity.y;
        else if (currentVelocity.y < -_velocity.y)
            currentVelocity.y = -_velocity.y;
    } else {
        if (currentVelocity.y > Scalar(0)) {
            if ((currentVelocity.y -= _acceleration.y * elapsedTimeSec) < Scalar(0))
                currentVelocity.y = Scalar(0);
        } else {
            if ((currentVelocity.y += _acceleration.y * elapsedTimeSec) > Scalar(0))
                currentVelocity.y = Scalar(0);
        }
    }

    if (direction.z != Scalar(0)) {
        currentVelocity.z += direction.z * _acceleration.z * elapsedTimeSec;

        if (currentVelocity.z > _velocity.z)
            currentVelocity.z = _velocity.z;
        else if (currentVelocity.z < -_velocity.z)
            currentVelocity.z = -_velocity.z;
    } else {
        if (currentVelocity.z > Scalar(0)) {
            if ((currentVelocity.z -= _acceleration.z * elapsedTimeSec) < Scalar(0))
                currentVelocity.z = Scalar(0);
        } else {
            if ((currentVelocity.z += _acceleration.z * elapsedTimeSec) > Scalar(0))
                currentVelocity.z = Scalar(0);
        }
    }
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::onResize(int width, int height) {
    perspective(static_cast<Scalar>(width) / static_cast<Scalar>(height));
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::setModel(const Mat4& model) {
    camera.model = model;
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags) const {
    vkCmdPushConstants(commandBuffer, layout.handle, stageFlags, 0, sizeof(Camera), &camera);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const Mat4& model, VkShaderStageFlags stageFlags) {
    const Camera aCamera{ .model = model, .view = camera.view, .proj = camera.proj };
    vkCmdPushConstants(commandBuffer, layout.handle, stageFlags, 0, sizeof(Camera), &aCamera);
}

template<typename Scalar>
const typename BaseCameraControllerT<Scalar>::Camera& BaseCameraControllerT<Scalar>::cam() const {
    return camera;
}

template<typename Scalar>
const typename BaseCameraControllerT<Scalar>::Camera& BaseCameraControllerT<Scalar>::previousCamera() const {
    return _previousCamera;
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::onPositionChanged() {
}

template<typename Scalar>
const typename BaseCameraControllerT<Scalar>::Quat& BaseCameraControllerT<Scalar>::getOrientation() const {
    return orientation;
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::setTargetYAxis(const Vec3& axis) {
    targetYAxis = axis;
}

template<typename Scalar>
const typename BaseCameraControllerT<Scalar>::Vec3& BaseCameraControllerT<Scalar>::getYAxis() {
    return yAxis;
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::newFrame() {
    _moved = false;
    _previousCamera = camera;
}

template<typename Scalar>
bool BaseCameraControllerT<Scalar>::moved() const {
    return _moved;
}

template<typename Scalar>
Scalar BaseCameraControllerT<Scalar>::near() const {
    return znear;
}

template<typename Scalar>
Scalar BaseCameraControllerT<Scalar>::far() const {
    return zfar;
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::fieldOfView(Scalar value) {
    perspective(value, aspectRatio, znear, zfar);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::jitter(Scalar jx, Scalar jy) {
    perspective(fov, aspectRatio, znear, zfar);
    Mat4 jMatrix = glm::translate(Mat4{1}, Vec3{jx, jy, Scalar(0)});
    camera.proj = jMatrix * camera.proj;
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::extract(Frustum& frustum) const {
    Frustum::extractFrustum(frustum, camera.proj * camera.view);
}

template<typename Scalar>
void BaseCameraControllerT<Scalar>::extractAABB(Vec3& bMin, Vec3& bMax) const {
    bMin = Vec3(std::numeric_limits<Scalar>::max());
    bMax = Vec3(std::numeric_limits<Scalar>::lowest());

    const auto near = znear;
    const auto far = zfar;
    const auto aspect = aspectRatio;
    const auto fovRad = glm::radians(fov);

    const auto inv_view = glm::inverse(camera.view);

    glm::vec<2, Scalar, glm::defaultp> nearCorner{Scalar(0), Scalar(0)};
    nearCorner.y = std::tan(fovRad / Scalar(2)) * -near;
    nearCorner.x = nearCorner.y * aspect;

    corners[0] = inv_view * Vec4(nearCorner, -near, Scalar(1));
    corners[1] = inv_view * Vec4(-nearCorner, -near, Scalar(1));

    nearCorner.y *= Scalar(-1);
    corners[2] = inv_view * Vec4(nearCorner, -near, Scalar(1));
    corners[3] = inv_view * Vec4(-nearCorner, -near, Scalar(1));

    glm::vec<2, Scalar, glm::defaultp> farCorner{Scalar(0), Scalar(0)};
    farCorner.y = std::tan(fovRad / Scalar(2)) * -far;
    farCorner.x = farCorner.y * aspect;

    corners[4] = inv_view * Vec4(farCorner, -far, Scalar(1));
    corners[5] = inv_view * Vec4(-farCorner, -far, Scalar(1));

    farCorner.y *= Scalar(-1);
    corners[6] = inv_view * Vec4(farCorner, -far, Scalar(1));
    corners[7] = inv_view * Vec4(-farCorner, -far, Scalar(1));

    for (auto& corner : corners) {
        corner /= corner.w;
        bMin = glm::min(corner.xyz(), bMin);
        bMax = glm::max(corner.xyz(), bMax);
    }
}

template<typename Scalar>
bool FrustumT<Scalar>::test(const Vec3& point) const {
    using namespace glm;
    const auto v = ClipPlane(point, Scalar(1));
    Scalar outside = Scalar(0);
    outside += step(dot(cp[LEFT_PLANE], v), Scalar(0)) + step(dot(cp[RIGHT_PLANE], v), Scalar(0));
    outside += step(dot(cp[BOTTOM_PLANE], v), Scalar(0)) + step(dot(cp[TOP_PLANE], v), Scalar(0));
    outside += step(dot(cp[NEAR_PLANE], v), Scalar(0)) + step(dot(cp[FAR_PLANE], v), Scalar(0));

    return outside == Scalar(0);
}

template<typename Scalar>
bool FrustumT<Scalar>::test(const Vec3& bMin, const Vec3& bMax) const {
    using Vec4 = glm::vec<4, Scalar, glm::defaultp>;
    using namespace glm;

    auto corners = std::array<Vec4, 8> {{
        Vec4(bMin.x, bMin.y, bMin.z, Scalar(1)), Vec4(bMax.x, bMin.y, bMin.z, Scalar(1)), Vec4(bMin.x, bMax.y, bMin.z, Scalar(1)),
        Vec4(bMax.x, bMax.y, bMin.z, Scalar(1)), Vec4(bMin.x, bMin.y, bMax.z, Scalar(1)), Vec4(bMax.x, bMin.y, bMax.z, Scalar(1)),
        Vec4(bMin.x, bMax.y, bMax.z, Scalar(1)), Vec4(bMax.x, bMax.y, bMax.z, Scalar(1))
    }};

    for (int i = 0; i < 6; ++i) {
        Scalar outside = Scalar(0);
        outside += step(dot(cp[i], corners[0]), Scalar(0));
        outside += step(dot(cp[i], corners[1]), Scalar(0));
        outside += step(dot(cp[i], corners[2]), Scalar(0));
        outside += step(dot(cp[i], corners[3]), Scalar(0));
        outside += step(dot(cp[i], corners[4]), Scalar(0));
        outside += step(dot(cp[i], corners[5]), Scalar(0));
        outside += step(dot(cp[i], corners[6]), Scalar(0));
        outside += step(dot(cp[i], corners[7]), Scalar(0));

        if (outside == Scalar(8)) return false;
    }

    return true;
}

template<typename Scalar>
bool FrustumT<Scalar>::test(const Vec3& boxCenter, Scalar scale) {
    using Vec4 = glm::vec<4, Scalar, glm::defaultp>;
    using namespace glm;
    static auto corners = std::array<Vec4, 8> {{
        Vec4(Scalar(-0.5), Scalar(-0.5), Scalar(-0.5), Scalar(0.5)), Vec4(Scalar(0.5), Scalar(-0.5), Scalar(-0.5), Scalar(0.5)),
        Vec4(Scalar(0.5), Scalar(-0.5), Scalar(0.5), Scalar(0.5)), Vec4(Scalar(-0.5), Scalar(-0.5), Scalar(0.5), Scalar(0.5)),
        Vec4(Scalar(-0.5), Scalar(0.5), Scalar(-0.5), Scalar(0.5)), Vec4(Scalar(0.5), Scalar(0.5), Scalar(-0.5), Scalar(0.5)),
        Vec4(Scalar(0.5), Scalar(0.5), Scalar(0.5), Scalar(0.5)), Vec4(Scalar(-0.5), Scalar(0.5), Scalar(0.5), Scalar(0.5)),
    }};

    const auto bc = Vec4(boxCenter, Scalar(0.5));
    const auto s = Vec4(scale, scale, scale, Scalar(1));
    for (int i = 0; i < 6; ++i) {
        Scalar outside = Scalar(0);
        outside += step(dot(cp[i], bc + corners[0] * s), Scalar(0));
        outside += step(dot(cp[i], bc + corners[1] * s), Scalar(0));
        outside += step(dot(cp[i], bc + corners[2] * s), Scalar(0));
        outside += step(dot(cp[i], bc + corners[3] * s), Scalar(0));
        outside += step(dot(cp[i], bc + corners[4] * s), Scalar(0));
        outside += step(dot(cp[i], bc + corners[5] * s), Scalar(0));
        outside += step(dot(cp[i], bc + corners[6] * s), Scalar(0));
        outside += step(dot(cp[i], bc + corners[7] * s), Scalar(0));

        if (outside == Scalar(8)) return false;
    }

    return true;
}

template<typename Scalar>
void FrustumT<Scalar>::extractFrustum(FrustumT& frustum, Mat4 M) {
    const auto m1 = glm::row(M, 0);
    const auto m4 = glm::row(M, 3);

    frustum.cp[LEFT_PLANE].x = m4[0] + m1[0];
    frustum.cp[LEFT_PLANE].y = m4[1] + m1[1];
    frustum.cp[LEFT_PLANE].z = m4[2] + m1[2];
    frustum.cp[LEFT_PLANE].w = m4[3] + m1[3];

    frustum.cp[RIGHT_PLANE].x = m4[0] - m1[0];
    frustum.cp[RIGHT_PLANE].y = m4[1] - m1[1];
    frustum.cp[RIGHT_PLANE].z = m4[2] - m1[2];
    frustum.cp[RIGHT_PLANE].w = m4[3] - m1[3];

    const auto m2 = glm::row(M, 1);

    frustum.cp[BOTTOM_PLANE].x = m4[0] + m2[0];
    frustum.cp[BOTTOM_PLANE].y = m4[1] + m2[1];
    frustum.cp[BOTTOM_PLANE].z = m4[2] + m2[2];
    frustum.cp[BOTTOM_PLANE].w = m4[3] + m2[3];

    frustum.cp[TOP_PLANE].x = m4[0] - m2[0];
    frustum.cp[TOP_PLANE].y = m4[1] - m2[1];
    frustum.cp[TOP_PLANE].z = m4[2] - m2[2];
    frustum.cp[TOP_PLANE].w = m4[3] - m2[3];

    const auto m3 = glm::row(M, 2);

    frustum.cp[NEAR_PLANE].x = m3[0];
    frustum.cp[NEAR_PLANE].y = m3[1];
    frustum.cp[NEAR_PLANE].z = m3[2];
    frustum.cp[NEAR_PLANE].w = m3[3];

    frustum.cp[FAR_PLANE].x = m4[0] - m3[0];
    frustum.cp[FAR_PLANE].y = m4[1] - m3[1];
    frustum.cp[FAR_PLANE].z = m4[2] - m3[2];
    frustum.cp[FAR_PLANE].w = m4[3] - m3[3];

    for (auto& p : frustum.cp) {
        auto invLength = glm::inversesqrt(glm::dot(p.xyz(), p.xyz()));
        p *= invLength;
    }
}

template struct FrustumT<float>;
template struct FrustumT<double>;
template struct BaseCameraControllerT<float>;
template struct BaseCameraControllerT<double>;
