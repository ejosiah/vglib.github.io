#pragma once

#include "camera_base.h"

template<typename Scalar>
struct FirstPersonSpectatorCameraSettingsT : public BaseCameraSettingsT<Scalar> {
};

template<typename Scalar>
class SpectatorCameraControllerT : public BaseCameraControllerT<Scalar> {
public:
    using Base = BaseCameraControllerT<Scalar>;
    using Settings = FirstPersonSpectatorCameraSettingsT<Scalar>;

    SpectatorCameraControllerT(InputManager& inputManager, const Settings& settings = {});

    void update(float elapsedTime) override;

    void rotate(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) override;
};

template<typename Scalar>
class FirstPersonCameraControllerT : public SpectatorCameraControllerT<Scalar> {
public:
    using Base = SpectatorCameraControllerT<Scalar>;
    using Vec3 = typename Base::Vec3;
    using Settings = FirstPersonSpectatorCameraSettingsT<Scalar>;

    FirstPersonCameraControllerT(InputManager& inputManager, const Settings& settings = {});

    void move(Scalar dx, Scalar dy, Scalar dz) override;
};

using FirstPersonSpectatorCameraSettings = FirstPersonSpectatorCameraSettingsT<float>;
using DoubleFirstPersonSpectatorCameraSettings = FirstPersonSpectatorCameraSettingsT<double>;

using SpectatorCameraController = SpectatorCameraControllerT<float>;
using DoubleSpectatorCameraController = SpectatorCameraControllerT<double>;

using FirstPersonCameraController = FirstPersonCameraControllerT<float>;
using DoubleFirstPersonCameraController = FirstPersonCameraControllerT<double>;
