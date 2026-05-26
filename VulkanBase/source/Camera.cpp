#include "Camera.h"

#include <memory>

namespace {
    template<typename Scalar>
    void configureBase(CameraSettingsT<Scalar>& settings);

    template<typename Scalar>
    void configureBase(const CameraSettingsT<Scalar>& settings, BaseCameraSettingsT<Scalar>& baseSettings);
}

template<typename Scalar>
CameraControllerT<Scalar>::CameraControllerT(InputManager& inputManager, const Settings& settings)
    : currentMode(settings.mode)
    , firstPerson(inputManager.mapToKey(Key::_1, "First Person mode", Action::detectInitialPressOnly()))
    , spectator(inputManager.mapToKey(Key::_2, "Spectator mode", Action::detectInitialPressOnly()))
    , flight(inputManager.mapToKey(Key::_3, "Flight mode", Action::detectInitialPressOnly()))
    , orbit(inputManager.mapToKey(Key::_4, "Orbit Person mode", Action::detectInitialPressOnly()))
{
    auto configuredSettings = settings;
    configureBase(configuredSettings);

    cameras[CameraMode::FIRST_PERSON] = std::make_unique<FirstPersonCameraControllerT<Scalar>>(inputManager, configuredSettings.firstPerson);
    cameras[CameraMode::SPECTATOR] = std::make_unique<SpectatorCameraControllerT<Scalar>>(inputManager, configuredSettings.firstPerson);
    cameras[CameraMode::FLIGHT] = std::make_unique<FlightCameraControllerT<Scalar>>(inputManager, configuredSettings.flight);
    cameras[CameraMode::ORBIT] = std::make_unique<OrbitingCameraControllerT<Scalar>>(inputManager, configuredSettings.orbit);
}

namespace {
    template<typename Scalar>
    void configureBase(CameraSettingsT<Scalar>& settings) {
        configureBase(settings, static_cast<BaseCameraSettingsT<Scalar>&>(settings.orbit));
        configureBase(settings, static_cast<BaseCameraSettingsT<Scalar>&>(settings.firstPerson));
        configureBase(settings, static_cast<BaseCameraSettingsT<Scalar>&>(settings.flight));
        configureBase(settings, static_cast<BaseCameraSettingsT<Scalar>&>(settings.orbit));
    }

    template<typename Scalar>
    void configureBase(const CameraSettingsT<Scalar>& settings, BaseCameraSettingsT<Scalar>& baseSettings) {
        baseSettings.acceleration = settings.acceleration;
        baseSettings.velocity = settings.velocity;
        baseSettings.rotationSpeed = settings.rotationSpeed;
        baseSettings.fieldOfView = settings.fieldOfView;
        baseSettings.aspectRatio = settings.aspectRatio;
        baseSettings.zNear = settings.zNear;
        baseSettings.zFar = settings.zFar;
        baseSettings.minZoom = settings.minZoom;
        baseSettings.maxZoom = settings.maxZoom;
        baseSettings.floorOffset = settings.floorOffset;
        baseSettings.handleZoom = settings.handleZoom;
        baseSettings.horizontalFov = settings.horizontalFov;
    }
}

template<typename Scalar>
void CameraControllerT<Scalar>::update(float time) {
    cameras[currentMode]->update(time);
}

template<typename Scalar>
void CameraControllerT<Scalar>::processInput() {
    if (firstPerson.isPressed()) {
        setMode(CameraMode::FIRST_PERSON);
    }
    if (spectator.isPressed()) {
        setMode(CameraMode::SPECTATOR);
    }
    if (flight.isPressed()) {
        setMode(CameraMode::FLIGHT);
    }
    if (orbit.isPressed()) {
        setMode(CameraMode::ORBIT);
    }

    cameras[currentMode]->processInput();
}

template<typename Scalar>
void CameraControllerT<Scalar>::lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    cameras[currentMode]->lookAt(eye, target, up);
}

template<typename Scalar>
void CameraControllerT<Scalar>::perspective(Scalar fovx, Scalar aspect, Scalar znear, Scalar zfar) {
    cameras[currentMode]->perspective(fovx, aspect, znear, zfar);
}

template<typename Scalar>
void CameraControllerT<Scalar>::perspective(Scalar aspect) {
    for (auto& [_, cam] : cameras) {
        cam->perspective(aspect);
    }
}

template<typename Scalar>
void CameraControllerT<Scalar>::rotateSmoothly(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) {
    cameras[currentMode]->rotateSmoothly(headingDegrees, pitchDegrees, rollDegrees);
}

template<typename Scalar>
void CameraControllerT<Scalar>::rotate(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) {
    cameras[currentMode]->rotate(headingDegrees, pitchDegrees, rollDegrees);
}

template<typename Scalar>
void CameraControllerT<Scalar>::move(Scalar dx, Scalar dy, Scalar dz) {
    cameras[currentMode]->move(dx, dy, dz);
}

template<typename Scalar>
void CameraControllerT<Scalar>::move(const Vec3& direction, const Vec3& amount) {
    cameras[currentMode]->move(direction, amount);
}

template<typename Scalar>
void CameraControllerT<Scalar>::position(const Vec3& pos) {
    cameras[currentMode]->position(pos);
}

template<typename Scalar>
void CameraControllerT<Scalar>::updatePosition(const Vec3& direction, Scalar elapsedTimeSec) {
    cameras[currentMode]->updatePosition(direction, elapsedTimeSec);
}

template<typename Scalar>
void CameraControllerT<Scalar>::undoRoll() {
    cameras[currentMode]->undoRoll();
}

template<typename Scalar>
void CameraControllerT<Scalar>::zoom(Scalar zoom, Scalar minZoom, Scalar maxZoom) {
    cameras[currentMode]->zoom(zoom, minZoom, maxZoom);
}

template<typename Scalar>
void CameraControllerT<Scalar>::onResize(int width, int height) {
    for (auto& [_, cam] : cameras) {
        cam->onResize(width, height);
    }
}

template<typename Scalar>
void CameraControllerT<Scalar>::setModel(const Mat4& model) {
    cameras[currentMode]->setModel(model);
}

template<typename Scalar>
void CameraControllerT<Scalar>::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const Mat4& model, VkShaderStageFlags stageFlags) {
    cameras[currentMode]->push(commandBuffer, layout, model, stageFlags);
}

template<typename Scalar>
void CameraControllerT<Scalar>::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags) const {
    UNUSED_VARIABLE(commandBuffer);
    UNUSED_VARIABLE(layout);
    UNUSED_VARIABLE(stageFlags);
//    cameras[currentMode]->push(commandBuffer, layout, stageFlags);
}

template<typename Scalar>
const typename CameraControllerT<Scalar>::Camera& CameraControllerT<Scalar>::cam() const {
    return cameras[currentMode]->cam();
}

template<typename Scalar>
std::string CameraControllerT<Scalar>::mode() const {
    switch (currentMode) {
        case CameraMode::FIRST_PERSON: return "First Person";
        case CameraMode::SPECTATOR: return "Spectator";
        case CameraMode::FLIGHT: return "Flight";
        case CameraMode::ORBIT: return "Orbit";
        default: return "Unknown";
    }
}

template<typename Scalar>
const typename CameraControllerT<Scalar>::Vec3& CameraControllerT<Scalar>::position() const {
    return cameras[currentMode]->position();
}

template<typename Scalar>
const typename CameraControllerT<Scalar>::Vec3& CameraControllerT<Scalar>::velocity() const {
    return cameras[currentMode]->velocity();
}

template<typename Scalar>
const typename CameraControllerT<Scalar>::Vec3& CameraControllerT<Scalar>::acceleration() const {
    return cameras[currentMode]->acceleration();
}

template<typename Scalar>
void CameraControllerT<Scalar>::setMode(CameraMode mode) {
    auto prevMode = currentMode;
    currentMode = mode;

    if (prevMode == CameraMode::NONE) {
        return;
    }

    auto newPos = cameras[prevMode]->position();
    switch (currentMode) {
        case CameraMode::FIRST_PERSON:
            newPos.y = cameras[currentMode]->position().y;
            cameras[currentMode]->position(newPos);
            break;
        case CameraMode::SPECTATOR:
        case CameraMode::FLIGHT:
            if (prevMode != CameraMode::ORBIT) {
                cameras[currentMode]->position(newPos);
            }
            break;
        case CameraMode::ORBIT: {
            auto yAxis = cameras[prevMode]->getYAxis();
            auto orbitCam = dynamic_cast<OrbitingCameraControllerT<Scalar> *>(cameras[currentMode].get());
            orbitCam->updateModel(newPos);
            orbitCam->setTargetYAxis(yAxis);
            orbitCam->position(newPos);
            orbitCam->rotate(Scalar(0), Scalar(-30), Scalar(0));
            auto pos = cameras[currentMode]->position();
            UNUSED_VARIABLE(pos);
            break;
        }
        case CameraMode::NONE:
            currentMode = CameraMode::SPECTATOR;
            break;
    }
}

template<typename Scalar>
bool CameraControllerT<Scalar>::isInFirstPersonMode() const {
    return currentMode == CameraMode::FIRST_PERSON;
}

template<typename Scalar>
bool CameraControllerT<Scalar>::isInFlightMode() const {
    return currentMode == CameraMode::FLIGHT;
}

template<typename Scalar>
bool CameraControllerT<Scalar>::isInSpectatorMode() const {
    return currentMode == CameraMode::SPECTATOR;
}

template<typename Scalar>
bool CameraControllerT<Scalar>::isInObitMode() const {
    return currentMode == CameraMode::ORBIT;
}

template<typename Scalar>
const typename CameraControllerT<Scalar>::Quat& CameraControllerT<Scalar>::getOrientation() const {
    return cameras[currentMode]->getOrientation();
}

template<typename Scalar>
void CameraControllerT<Scalar>::newFrame() {
    cameras[currentMode]->newFrame();
}

template<typename Scalar>
bool CameraControllerT<Scalar>::moved() const {
    return cameras[currentMode]->moved();
}

template<typename Scalar>
Scalar CameraControllerT<Scalar>::near() const {
    return cameras[currentMode]->near();
}

template<typename Scalar>
Scalar CameraControllerT<Scalar>::far() const {
    return cameras[currentMode]->far();
}

template<typename Scalar>
void CameraControllerT<Scalar>::fieldOfView(Scalar value) {
    return cameras[currentMode]->fieldOfView(value);
}

template<typename Scalar>
const typename CameraControllerT<Scalar>::Camera& CameraControllerT<Scalar>::previousCamera() const {
    return cameras[currentMode]->previousCamera();
}

template<typename Scalar>
void CameraControllerT<Scalar>::jitter(Scalar jx, Scalar jy) {
    return cameras[currentMode]->jitter(jx, jy);
}

template<typename Scalar>
void CameraControllerT<Scalar>::extract(Frustum& frustum) const {
    cameras[currentMode]->extract(frustum);
}

template<typename Scalar>
void CameraControllerT<Scalar>::extractAABB(Vec3& bMin, Vec3& bMax) const {
    cameras[currentMode]->extractAABB(bMin, bMax);
}

template class CameraControllerT<float>;
template class CameraControllerT<double>;
