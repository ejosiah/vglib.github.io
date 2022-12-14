#include "FlightCamera.h"

FlightCameraController::FlightCameraController(InputManager &inputManager, const FlightCameraSettings &settings)
: BaseCameraController(inputManager, settings)
, YawSpeed(settings.yawSpeed)
{

}

void FlightCameraController::update(float elapsedTime) {
    float dx = -direction.x * YawSpeed * elapsedTime;
    float dy = mouse.relativePosition.y;
    float dz = -mouse.relativePosition.x;
    rotateSmoothly(0.0f, dy, dz);

    if (dx != 0.0f) {
        rotate(dx, 0.0f, 0.0f);
    }

    direction.x = 0.0f; // ignore yaw motion when updating camera's velocity;
    updatePosition(direction, elapsedTime);
}

void FlightCameraController::rotate(float headingDegrees, float pitchDegrees, float rollDegrees) {
    if(headingDegrees == 0 && pitchDegrees == 0 && rollDegrees == 0){
        return;
    }

    accumPitchDegrees += pitchDegrees;

    if (accumPitchDegrees > 360.0f)
        accumPitchDegrees -= 360.0f;

    if (accumPitchDegrees < -360.0f)
        accumPitchDegrees += 360.0f;

    glm::quat rot = glm::quat({ glm::radians(pitchDegrees), glm::radians(headingDegrees), glm::radians(rollDegrees) });
    orientation = rot * orientation;

    updateViewMatrix();
}
