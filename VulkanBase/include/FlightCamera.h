#pragma once

#include "camera_base.h"

constexpr float DEFAULT_YAW_SPEED = 100.0F;

template<typename Scalar>
struct FlightCameraSettingsT : BaseCameraSettingsT<Scalar> {
    Scalar yawSpeed = Scalar(DEFAULT_YAW_SPEED);
};

template<typename Scalar>
class FlightCameraControllerT : public BaseCameraControllerT<Scalar> {
public:
    using Settings = FlightCameraSettingsT<Scalar>;

    FlightCameraControllerT(InputManager& inputManager, const Settings& settings = {});

    void update(float elapsedTime) override;

    void rotate(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) override;

private:
    Scalar YawSpeed;
};

using FlightCameraSettings = FlightCameraSettingsT<float>;
using DoubleFlightCameraSettings = FlightCameraSettingsT<double>;

using FlightCameraController = FlightCameraControllerT<float>;
using DoubleFlightCameraController = FlightCameraControllerT<double>;
