#include "FirstPersonCamera.h"

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
SpectatorCameraControllerT<Scalar>::SpectatorCameraControllerT(InputManager& inputManager, const Settings& settings)
    : Base(inputManager, settings)
{
}

template<typename Scalar>
void SpectatorCameraControllerT<Scalar>::update(float elapsedTime) {
    const Scalar dx = -static_cast<Scalar>(this->mouse.relativePosition.x);
    const Scalar dy = -static_cast<Scalar>(this->mouse.relativePosition.y);
    this->rotateSmoothly(dx, dy, Scalar(0));
    this->updatePosition(this->direction, static_cast<Scalar>(elapsedTime));
}

template<typename Scalar>
void SpectatorCameraControllerT<Scalar>::rotate(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) {
    if (headingDegrees == Scalar(0) && pitchDegrees == Scalar(0) && rollDegrees == Scalar(0)) {
        return;
    }

    this->accumPitchDegrees += pitchDegrees;

    static std::vector<Scalar> pRots;
    if (pitchDegrees != Scalar(0)) pRots.push_back(pitchDegrees);

    if (this->accumPitchDegrees > Scalar(90)) {
        pitchDegrees = Scalar(90) - (this->accumPitchDegrees - pitchDegrees);
        this->accumPitchDegrees = Scalar(90);
    }

    if (this->accumPitchDegrees < Scalar(-90)) {
        pitchDegrees = Scalar(-90) - (this->accumPitchDegrees - pitchDegrees);
        this->accumPitchDegrees = Scalar(-90);
    }

    glm::qua<Scalar, glm::defaultp> rot;

    if (headingDegrees != Scalar(0)) {
        rot = from_axis_angle(WORLD_YAXIS_T<Scalar>, headingDegrees);
        this->orientation = this->orientation * rot;
    }

    if (pitchDegrees != Scalar(0)) {
        rot = from_axis_angle(WORLD_XAXIS_T<Scalar>, pitchDegrees);
        this->orientation = rot * this->orientation;
    }
    this->updateViewMatrix();
}

template<typename Scalar>
FirstPersonCameraControllerT<Scalar>::FirstPersonCameraControllerT(InputManager& inputManager, const Settings& settings)
    : Base(inputManager, settings)
{
}

template<typename Scalar>
void FirstPersonCameraControllerT<Scalar>::move(Scalar dx, Scalar dy, Scalar dz) {
    if (dx == Scalar(0) && dy == Scalar(0) && dz == Scalar(0)) return;
    Vec3 eyes = this->eyes;

    Vec3 forwards = normalize(cross(WORLD_YAXIS_T<Scalar>, this->xAxis));

    eyes += this->xAxis * dx;
    eyes += WORLD_YAXIS_T<Scalar> * dy;
    eyes += forwards * dz;

    this->position(eyes);
}

template class SpectatorCameraControllerT<float>;
template class SpectatorCameraControllerT<double>;
template class FirstPersonCameraControllerT<float>;
template class FirstPersonCameraControllerT<double>;
