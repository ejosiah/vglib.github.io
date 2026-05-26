#include "FlightCamera.h"

template<typename Scalar>
FlightCameraControllerT<Scalar>::FlightCameraControllerT(InputManager& inputManager, const Settings& settings)
    : BaseCameraControllerT<Scalar>(inputManager, settings)
    , YawSpeed(settings.yawSpeed)
{
}

template<typename Scalar>
void FlightCameraControllerT<Scalar>::update(float elapsedTime) {
    Scalar dx = -this->direction.x * YawSpeed * static_cast<Scalar>(elapsedTime);
    Scalar dy = static_cast<Scalar>(this->mouse.relativePosition.y);
    Scalar dz = -static_cast<Scalar>(this->mouse.relativePosition.x);
    this->rotateSmoothly(Scalar(0), dy, dz);

    if (dx != Scalar(0)) {
        rotate(dx, Scalar(0), Scalar(0));
    }

    this->direction.x = Scalar(0);
    this->updatePosition(this->direction, static_cast<Scalar>(elapsedTime));
}

template<typename Scalar>
void FlightCameraControllerT<Scalar>::rotate(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) {
    if (headingDegrees == Scalar(0) && pitchDegrees == Scalar(0) && rollDegrees == Scalar(0)) {
        return;
    }

    this->accumPitchDegrees += pitchDegrees;

    if (this->accumPitchDegrees > Scalar(360))
        this->accumPitchDegrees -= Scalar(360);

    if (this->accumPitchDegrees < Scalar(-360))
        this->accumPitchDegrees += Scalar(360);

    glm::qua<Scalar, glm::defaultp> rot = glm::qua<Scalar, glm::defaultp>({
        glm::radians(pitchDegrees),
        glm::radians(headingDegrees),
        glm::radians(rollDegrees)
    });
    this->orientation = rot * this->orientation;

    this->updateViewMatrix();
}

template class FlightCameraControllerT<float>;
template class FlightCameraControllerT<double>;
