#include <OrbitingCamera.h>
#include "OrbitingCamera.h"

OrbitingCameraController::OrbitingCameraController(InputManager& inputManager,  const OrbitingCameraSettings& settings)
: BaseCameraController(inputManager, settings)
, offsetDistance(settings.offsetDistance)
, orbitRollSpeed(settings.orbitRollSpeed)
, preferTargetYAxisOrbiting(settings.preferTargetYAxisOrbiting)
{
    minZoom = settings.orbitMinZoom;
    maxZoom = settings.orbitMaxZoom;
    offsetDistance = settings.offsetDistance;
    floorOffset = settings.modelHeight * 0.5f;
    handleZoom = false;
    model.position = {0.0f, floorOffset, 0.0f};
    model.orientation = glm::inverse(orientation);

    glm::vec3 target = model.position;
    if(!glm::any(glm::isnan(settings.target))){
        target = settings.target;
    }

    auto eyes = target + zAxis * offsetDistance;
    lookAt(eyes, target, targetYAxis);
}

void OrbitingCameraController::update(float elapsedTime) {
    float dx = mouse.relativePosition.x;
    float dy = mouse.relativePosition.y;

    rotateSmoothly(dx, dy, 0.0f);

    if (!preferTargetYAxisOrbiting) {
        float dz = direction.x * orbitRollSpeed * elapsedTime;
        if (dz != 0.0f) {
            rotate(0.0f, 0.0f, dz);
        }
    }

    if (zoomAmount != 0.0f) {
        zoom(zoomAmount, minZoom, maxZoom);
    }
}

void OrbitingCameraController::move(float dx, float dy, float dz) {
    // Operation Not supported
}

void OrbitingCameraController::move(const glm::vec3 &direction, const glm::vec3 &amount) {
    // Operation Not supported
}

void OrbitingCameraController::undoRoll() {
    lookAt(eyes, target, targetYAxis);
}

void OrbitingCameraController::zoom(float zoom, float minZoom, float maxZoom) {
    // Moves the Camera closer to or further away from the orbit
    // target. The zoom amounts are in world units.

    this->maxZoom = maxZoom;
    this->minZoom = minZoom;

    glm::vec3 offset = eyes - target;

    offsetDistance = glm::length(offset);
    offset = normalize(offset);
    offsetDistance += zoom;
    offsetDistance = std::min(std::max(offsetDistance, minZoom), maxZoom);

    offset *= offsetDistance;
    eyes = offset + target;

    updateViewMatrix();
}

void OrbitingCameraController::rotate(float headingDegrees, float pitchDegrees, float rollDegrees) {
    if(headingDegrees == 0 && pitchDegrees == 0 && rollDegrees == 0){
        return;
    }

    // Implements the rotation logic for the orbit style Camera mode.
    // Roll is ignored for target Y axis orbiting.
    //
    // Briefly here's how this orbit Camera implementation works. Switching to
    // the orbit Camera mode via the setBehavior() method will set the
    // Camera's orientation to match the orbit target's orientation. Calls to
    // rotateOrbit() will rotate this orientation. To turn this into a third
    // person style view the updateViewMatrix() method will move the Camera
    // position back 'orbitOffsetDistance' world units along the Camera's
    // local z axis from the orbit target's world position.
    pitchDegrees = -pitchDegrees;
    headingDegrees = -headingDegrees;
    rollDegrees = -rollDegrees;

    using namespace glm;
    glm::quat rot;

    if (preferTargetYAxisOrbiting)
    {
        if (headingDegrees != 0.0f)
        {
            rot = fromAxisAngle(targetYAxis, headingDegrees);
            orientation =  orientation * rot;
        }

        if (pitchDegrees != 0.0f)
        {
            rot = fromAxisAngle(WORLD_XAXIS, pitchDegrees);
            orientation = rot * orientation;
        }
    }
    else
    {
        rot = glm::quat({ radians(pitchDegrees), radians(headingDegrees), radians(rollDegrees) });
        orientation = rot * orientation;
    }
    updateViewMatrix();
}

void OrbitingCameraController::updateModel(const glm::vec3& position, const glm::quat& orientation) {
    model.orientation = glm::inverse(orientation);
    model.position = position;
}

void OrbitingCameraController::updateModel(const glm::vec3& bMin, const glm::vec3& bMax) {
    const auto dim = bMax - bMin;
    const auto center = (bMin + bMax) * 0.5f;
    model.position = center;
    target = center;
    offsetDistance = glm::length(dim);
    auto eyes = target + zAxis * offsetDistance;
    lookAt(eyes, target, targetYAxis);

}

void OrbitingCameraController::updateViewMatrix() {
    auto& view = camera.view;
    view = glm::mat4_cast(orientation);

    xAxis = glm::vec3(glm::row(view, 0));
    yAxis = glm::vec3(glm::row(view, 1));
    zAxis = glm::vec3(glm::row(view, 2));
    viewDir = -zAxis;

    eyes = target + zAxis * offsetDistance;

    view[3][0] = -dot(xAxis, eyes);
    view[3][1] = -dot(yAxis, eyes);
    view[3][2] =  -dot(zAxis, eyes);
    _moved = true;
}

void OrbitingCameraController::onPositionChanged() {
    auto newEyes = eyes + zAxis * offsetDistance;
    auto newTarget = eyes;
    lookAt(newEyes, newTarget, targetYAxis);
}

void OrbitingCameraController::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags) const {
    camera.model = getModel();
    BaseCameraController::push(commandBuffer, layout, stageFlags);
}

void OrbitingCameraController::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const glm::mat4 &model, VkShaderStageFlags stageFlags) {
    BaseCameraController::push(commandBuffer, layout, model, stageFlags);
}

glm::mat4 OrbitingCameraController::getModel() const {
    return glm::mat4_cast(model.orientation) * translate(glm::mat4(1), model.position);
}
