#include <OrbitingCamera.h>
#include "OrbitingCamera.h"

namespace {
    template<typename Scalar>
    glm::qua<Scalar, glm::defaultp> from_axis_angle(const glm::vec<3, Scalar, glm::defaultp>& axis, Scalar angle) {
        const auto halfAngle = glm::radians(angle) / Scalar(2);
        const auto w = std::cos(halfAngle);
        const auto xyz = axis * std::sin(halfAngle);
        return glm::qua<Scalar, glm::defaultp>(w, xyz);
    }
}

template<typename Scalar>
OrbitingCameraControllerT<Scalar>::OrbitingCameraControllerT(InputManager& inputManager, const Settings& settings)
    : Base(inputManager, settings)
    , offsetDistance(settings.offsetDistance)
    , orbitRollSpeed(settings.orbitRollSpeed)
    , preferTargetYAxisOrbiting(settings.preferTargetYAxisOrbiting)
{
    this->minZoom = settings.orbitMinZoom;
    this->maxZoom = settings.orbitMaxZoom;
    offsetDistance = settings.offsetDistance;
    this->floorOffset = settings.modelHeight * Scalar(0.5);
    this->handleZoom = false;
    model.position = {Scalar(0), this->floorOffset, Scalar(0)};
    model.orientation = glm::inverse(this->orientation);

    Vec3 target = model.position;
    if (!glm::any(glm::isnan(settings.target))) {
        target = settings.target;
    }

    auto eyes = target + this->zAxis * offsetDistance;
    this->lookAt(eyes, target, this->targetYAxis);
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::update(float elapsedTime) {
    const Scalar dx = static_cast<Scalar>(this->mouse.relativePosition.x);
    const Scalar dy = static_cast<Scalar>(this->mouse.relativePosition.y);

    this->rotateSmoothly(dx, dy, Scalar(0));

    if (!preferTargetYAxisOrbiting) {
        Scalar dz = this->direction.x * orbitRollSpeed * static_cast<Scalar>(elapsedTime);
        if (dz != Scalar(0)) {
            rotate(Scalar(0), Scalar(0), dz);
        }
    }

    if (this->zoomAmount != Scalar(0)) {
        zoom(this->zoomAmount, this->minZoom, this->maxZoom);
    }
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::move(Scalar dx, Scalar dy, Scalar dz) {
    UNUSED_VARIABLE(dx);
    UNUSED_VARIABLE(dy);
    UNUSED_VARIABLE(dz);
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::move(const Vec3& direction, const Vec3& amount) {
    UNUSED_VARIABLE(direction);
    UNUSED_VARIABLE(amount);
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::undoRoll() {
    this->lookAt(this->eyes, this->target, this->targetYAxis);
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::zoom(Scalar zoom, Scalar minZoom, Scalar maxZoom) {
    this->maxZoom = maxZoom;
    this->minZoom = minZoom;

    Vec3 offset = this->eyes - this->target;

    offsetDistance = glm::length(offset);
    offset = normalize(offset);
    offsetDistance += zoom;
    offsetDistance = std::min(std::max(offsetDistance, minZoom), maxZoom);

    offset *= offsetDistance;
    this->eyes = offset + this->target;

    updateViewMatrix();
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::rotate(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) {
    if (headingDegrees == Scalar(0) && pitchDegrees == Scalar(0) && rollDegrees == Scalar(0)) {
        return;
    }

    pitchDegrees = -pitchDegrees;
    headingDegrees = -headingDegrees;
    rollDegrees = -rollDegrees;

    glm::qua<Scalar, glm::defaultp> rot;

    if (preferTargetYAxisOrbiting) {
        if (headingDegrees != Scalar(0)) {
            rot = from_axis_angle(this->targetYAxis, headingDegrees);
            this->orientation = this->orientation * rot;
        }

        if (pitchDegrees != Scalar(0)) {
            rot = from_axis_angle(WORLD_XAXIS_T<Scalar>, pitchDegrees);
            this->orientation = rot * this->orientation;
        }
    } else {
        rot = glm::qua<Scalar, glm::defaultp>({glm::radians(pitchDegrees), glm::radians(headingDegrees), glm::radians(rollDegrees)});
        this->orientation = rot * this->orientation;
    }
    updateViewMatrix();
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::updateModel(const Vec3& position, const Quat& orientation) {
    model.orientation = glm::inverse(orientation);
    model.position = position;
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::updateModel(const Vec3& bMin, const Vec3& bMax) {
    const auto dim = bMax - bMin;
    const auto center = (bMin + bMax) * Scalar(0.5);
    model.position = center;
    this->target = center;
    offsetDistance = glm::length(dim);
    auto eyes = this->target + this->zAxis * offsetDistance;
    this->lookAt(eyes, this->target, this->targetYAxis);
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::updateViewMatrix() {
    auto& view = this->camera.view;
    view = glm::mat4_cast(this->orientation);

    this->xAxis = Vec3(glm::row(view, 0));
    this->yAxis = Vec3(glm::row(view, 1));
    this->zAxis = Vec3(glm::row(view, 2));
    this->viewDir = -this->zAxis;

    this->eyes = this->target + this->zAxis * offsetDistance;

    view[3][0] = -dot(this->xAxis, this->eyes);
    view[3][1] = -dot(this->yAxis, this->eyes);
    view[3][2] = -dot(this->zAxis, this->eyes);
    this->_moved = true;
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::onPositionChanged() {
    auto newEyes = this->eyes + this->zAxis * offsetDistance;
    auto newTarget = this->eyes;
    this->lookAt(newEyes, newTarget, this->targetYAxis);
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags) const {
    this->camera.model = getModel();
    Base::push(commandBuffer, layout, stageFlags);
}

template<typename Scalar>
void OrbitingCameraControllerT<Scalar>::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const Mat4& model, VkShaderStageFlags stageFlags) {
    Base::push(commandBuffer, layout, model, stageFlags);
}

template<typename Scalar>
typename OrbitingCameraControllerT<Scalar>::Mat4 OrbitingCameraControllerT<Scalar>::getModel() const {
    return glm::mat4_cast(model.orientation) * glm::translate(Mat4(1), model.position);
}

template class OrbitingCameraControllerT<float>;
template class OrbitingCameraControllerT<double>;
