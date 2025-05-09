#include "camera_base.h"
#include "AbstractCamera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>

BaseCameraController::BaseCameraController(InputManager& inputManager, const BaseCameraSettings& settings)
    : fov(settings.fieldOfView)
    , aspectRatio(settings.aspectRatio)
    , znear(settings.zNear)
    , zfar(settings.zFar)
    , minZoom(settings.minZoom)
    , maxZoom(settings.maxZoom)
    , rotationSpeed(settings.rotationSpeed)
    , accumPitchDegrees(0.0f)
    , floorOffset(settings.floorOffset)
    , handleZoom(settings.handleZoom)
    , horizontalFov(settings.horizontalFov)
    , eyes(0.0f)
    , target(0.0f)
    , targetYAxis(0.0f, 1.0f, 0.0f)
    , xAxis(1.0f, 0.0f, 0.0f)
    , yAxis(0.0f, 1.0f, 0.0f)
    , zAxis(0.0f, 0.0f, 1.0f)
    , viewDir(0.0f, 0.0f, -1.0f)
    , _acceleration(settings.acceleration)
    , _velocity(settings.velocity)
    , currentVelocity(0)
    , orientation(1, 0, 0, 0)
    , direction(0)
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
        position({0.0f, floorOffset, 0.0f});
        perspective(fov, aspectRatio, znear, zfar);
    }

void BaseCameraController::processInput() {
    processMovementInput();
    processZoomInput();
}

void BaseCameraController::processMovementInput() {
    direction = glm::vec3(0);
    auto vel = currentVelocity;
    if(_move.forward->isPressed()){
        currentVelocity.x = vel.x;
        currentVelocity.y = vel.y;
        currentVelocity.z = 0;
    }else if(_move.forward->isHeld()){
        direction.z += 1.0f;
    }

    if(_move.back->isPressed()){
        currentVelocity.x = vel.x;
        currentVelocity.y = vel.y;
        currentVelocity.z = 0;
    }else if(_move.back->isHeld()){
        direction.z -= 1.0f;
    }

    if(_move.right->isPressed()){
        currentVelocity.x = 0;
        currentVelocity.y = vel.y;
        currentVelocity.z = vel.z;
    }else if(_move.right->isHeld()){
        direction.x += 1.0f;
    }

    if(_move.left->isPressed()){
        currentVelocity.x = 0;
        currentVelocity.y = vel.y;
        currentVelocity.z = vel.z;
    }else if(_move.left->isHeld()){
        direction.x -= 1.0f;
    }


    if(_move.up->isPressed()){
        currentVelocity.x = vel.x;
        currentVelocity.y = 0.0f;
        currentVelocity.z = vel.z;
    }else if(_move.up->isHeld()){
        direction.y += 1.0f;
    }

    if(_move.down->isPressed()){
        currentVelocity.x = vel.x;
        currentVelocity.y = 0.0f;
        currentVelocity.z = vel.z;
    }else if(_move.down->isHeld()){
        direction.y -= 1.0f;
    }
}

void BaseCameraController::processZoomInput() {
    zoomAmount = 0.0f;
    if(zoomIn.isPressed()){
        zoomAmount = -zoomDelta;
    }else if(zoomOut.isPressed()){
        zoomAmount = zoomDelta;
    }
    if(handleZoom && zoomAmount != 0){
        zoom(zoomAmount, minZoom, maxZoom);
    }
}


void BaseCameraController::lookAt(const glm::vec3 &eye, const glm::vec3 &target, const glm::vec3 &up) {

    this->eyes = eye;
    this->target = target;

    auto& view = camera.view;
    view = glm::lookAt(eye, target, up);
    // Extract the pitch angle from the view matrix.
    accumPitchDegrees = glm::degrees(asinf(view[1][2]));	// TODO change this matrix is colomn matrix

    xAxis = glm::vec3(row(view, 0));
    yAxis = glm::vec3(row(view, 1));
    zAxis = glm::vec3(row(view, 2));

    viewDir = -zAxis;

    accumPitchDegrees = glm::degrees(asinf(view[1][2]));

    orientation = glm::quat(view);
    updateViewMatrix();
}

void BaseCameraController::perspective(float aspect) {
    perspective(fov, aspect, znear, zfar);
}

void BaseCameraController::perspective(float fov, float aspect, float znear, float zfar) {
    camera.proj = vkn::perspective(glm::radians(fov), aspect, znear, zfar, horizontalFov);
    this->fov = fov;
    aspectRatio = aspect;
    this->znear = znear;
    this->zfar = zfar;
    _moved = true;
}

void BaseCameraController::rotateSmoothly(float headingDegrees, float pitchDegrees, float rollDegrees) {
    headingDegrees *= rotationSpeed;
    pitchDegrees *= rotationSpeed;
    rollDegrees *= rotationSpeed;

    rotate(headingDegrees, pitchDegrees, rollDegrees);
}

void BaseCameraController::undoRoll() {
    lookAt(eyes, eyes + viewDir, WORLD_YAXIS);
}

void BaseCameraController::zoom(float zoom, float minZoom, float maxZoom) {
    zoom = std::min(std::max(zoom, minZoom), maxZoom);
    perspective(zoom, aspectRatio, znear, zfar);
}

void BaseCameraController::move(float dx, float dy, float dz) {
    if(dx == 0 && dy == 0 && dz == 0) return;   // TODO use close enough

    glm::vec3 eyes = this->eyes;
    glm::vec3 forwards = viewDir;

    eyes += xAxis * dx;
    eyes += WORLD_YAXIS * dy;
    eyes += forwards * dz;

    position(eyes);
}

void BaseCameraController::move(const glm::vec3 &direction, const glm::vec3 &amount) {
    eyes.x += direction.x * amount.x;
    eyes.y += direction.y * amount.y;
    eyes.z += direction.z * amount.z;

    updateViewMatrix();
}


void BaseCameraController::position(const glm::vec3 &pos) {
    eyes = pos;
    onPositionChanged();
    updateViewMatrix();
}

const glm::vec3& BaseCameraController::position() const {
    return eyes;
}

const glm::vec3& BaseCameraController::velocity() const {
    return currentVelocity;
}

const glm::vec3& BaseCameraController::acceleration() const {
    return _acceleration;
}

void BaseCameraController::updatePosition(const glm::vec3 &direction, float elapsedTimeSec) {
    // Moves the Camera using Newton's second law of motion. Unit mass is
    // assumed here to somewhat simplify the calculations. The direction vector
    // is in the range [-1,1].
    using namespace glm;
    if (dot(currentVelocity, currentVelocity) != 0.0f)
    {
        // Only move the Camera if the _velocity vector is not of zero length.
        // Doing this guards against the Camera slowly creeping around due to
        // floating point rounding errors.

        glm::vec3 displacement = (currentVelocity * elapsedTimeSec) +
                                 (0.5f * _acceleration * elapsedTimeSec * elapsedTimeSec);

        // Floating point rounding errors will slowly accumulate and cause the
        // Camera to move along each axis. To prevent any unintended movement
        // the displacement vector is clamped to zero for each direction that
        // the Camera isn't moving in. Note that the updateVelocity() method
        // will slowly decelerate the Camera's _velocity back to a stationary
        // state when the Camera is no longer moving along that direction. To
        // account for this the Camera's current _velocity is also checked.

        if (direction.x == 0.0f && closeEnough(currentVelocity.x, 0.0f))
            displacement.x = 0.0f;

        if (direction.y == 0.0f && closeEnough(currentVelocity.y, 0.0f))
            displacement.y = 0.0f;

        if (direction.z == 0.0f && closeEnough(currentVelocity.z, 0.0f))
            displacement.z = 0.0f;

        move(displacement.x, displacement.y, displacement.z);
    }

    // Continuously update the Camera's _velocity vector even if the Camera
    // hasn't moved during this call. When the Camera is no longer being moved
    // the Camera is decelerating back to its stationary state.

    updateVelocity(direction, elapsedTimeSec);
}

void BaseCameraController::updateViewMatrix() {
    auto& view = camera.view;
    view = glm::mat4_cast(orientation);

    xAxis = glm::vec3(glm::row(view, 0));
    yAxis = glm::vec3(glm::row(view, 1));
    zAxis = glm::vec3(glm::row(view, 2));
    viewDir = -zAxis;

    view[3][0] = -dot(xAxis, eyes);
    view[3][1] = -dot(yAxis, eyes);
    view[3][2] =  -dot(zAxis, eyes);
    _moved = true;
}

void BaseCameraController::updateVelocity(const glm::vec3 &direction, float elapsedTimeSec) {
    // Updates the Camera's _velocity based on the supplied movement direction
    // and the elapsed time (since this method was last called). The movement
    // direction is in the range [-1,1].

    if (direction.x != 0.0f)
    {
        // Camera is moving along the x axis.
        // Linearly accelerate up to the Camera's max speed.

        currentVelocity.x += direction.x * _acceleration.x * elapsedTimeSec;

        if (currentVelocity.x > _velocity.x)
            currentVelocity.x = _velocity.x;
        else if (currentVelocity.x < -_velocity.x)
            currentVelocity.x = -_velocity.x;
    }
    else
    {
        // Camera is no longer moving along the x axis.
        // Linearly decelerate back to stationary state.

        if (currentVelocity.x > 0.0f)
        {
            if ((currentVelocity.x -= _acceleration.x * elapsedTimeSec) < 0.0f)
                currentVelocity.x = 0.0f;
        }
        else
        {
            if ((currentVelocity.x += _acceleration.x * elapsedTimeSec) > 0.0f)
                currentVelocity.x = 0.0f;
        }
    }

    if (direction.y != 0.0f)
    {
        // Camera is moving along the y axis.
        // Linearly accelerate up to the Camera's max speed.

        currentVelocity.y += direction.y * _acceleration.y * elapsedTimeSec;

        if (currentVelocity.y > _velocity.y)
            currentVelocity.y = _velocity.y;
        else if (currentVelocity.y < -_velocity.y)
            currentVelocity.y = -_velocity.y;
    }
    else
    {
        // Camera is no longer moving along the y axis.
        // Linearly decelerate back to stationary state.

        if (currentVelocity.y > 0.0f)
        {
            if ((currentVelocity.y -= _acceleration.y * elapsedTimeSec) < 0.0f)
                currentVelocity.y = 0.0f;
        }
        else
        {
            if ((currentVelocity.y += _acceleration.y * elapsedTimeSec) > 0.0f)
                currentVelocity.y = 0.0f;
        }
    }

    if (direction.z != 0.0f)
    {
        // Camera is moving along the z axis.
        // Linearly accelerate up to the Camera's max speed.

        currentVelocity.z += direction.z * _acceleration.z * elapsedTimeSec;

        if (currentVelocity.z > _velocity.z)
            currentVelocity.z = _velocity.z;
        else if (currentVelocity.z < -_velocity.z)
            currentVelocity.z = -_velocity.z;
    }
    else
    {
        // Camera is no longer moving along the z axis.
        // Linearly decelerate back to stationary state.

        if (currentVelocity.z > 0.0f)
        {
            if ((currentVelocity.z -= _acceleration.z * elapsedTimeSec) < 0.0f)
                currentVelocity.z = 0.0f;
        }
        else
        {
            if ((currentVelocity.z += _acceleration.z * elapsedTimeSec) > 0.0f)
                currentVelocity.z = 0.0f;
        }
    }
}


void BaseCameraController::onResize(int width, int height) {
    perspective(static_cast<float>(width)/static_cast<float>(height));
}

void BaseCameraController::setModel(const glm::mat4& model) {
    camera.model = model;
}

void BaseCameraController::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags) const {
    vkCmdPushConstants(commandBuffer, layout.handle, stageFlags, 0, sizeof(Camera), &camera);
}

void BaseCameraController::push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const glm::mat4& model, VkShaderStageFlags stageFlags) {
    const Camera aCamera{ .model = model, .view = camera.view, .proj = camera.proj };
    vkCmdPushConstants(commandBuffer, layout.handle, stageFlags, 0, sizeof(Camera), &aCamera);
}

const Camera &BaseCameraController::cam() const {
    return camera;
}

const Camera &BaseCameraController::previousCamera() const {
    return _previousCamera;
}


void BaseCameraController::onPositionChanged() {

}

const glm::quat &BaseCameraController::getOrientation() const {
    return orientation;
}

void BaseCameraController::setTargetYAxis(const glm::vec3& axis) {
    targetYAxis = axis;
}

const glm::vec3 &BaseCameraController::getYAxis() {
    return yAxis;
}

void BaseCameraController::newFrame() {
    _moved = false;
    _previousCamera = camera;
}

bool BaseCameraController::moved() const {
    return _moved;
}

float BaseCameraController::near() const {
    return znear;
}

float BaseCameraController::far() const {
    return zfar;
}

void BaseCameraController::fieldOfView(float value) {
    perspective(value, aspectRatio, znear, zfar);
}

void BaseCameraController::jitter(float jx, float jy) {
    perspective(fov, aspectRatio, znear, zfar);
    glm::mat4 jMatrix = glm::translate(glm::mat4{1}, {jx, jy, 0});
    camera.proj = jMatrix * camera.proj;
}

void BaseCameraController::extract(Frustum &frustum) const {
    Frustum::extractFrustum(frustum, camera.proj * camera.view);
}

void BaseCameraController::extractAABB(glm::vec3 &bMin, glm::vec3 &bMax) const {
    bMin = glm::vec3(MAX_FLOAT);
    bMax = glm::vec3(MIN_FLOAT);

    const auto near = znear;
    const auto far = zfar;
    const auto aspect = aspectRatio;
    const auto fovRad = glm::radians(fov);

    const auto inv_view = glm::inverse(camera.view);

    glm::vec2 nearCorner{0, 0};
    nearCorner.y = glm::tan(fovRad / 2) * -near; // TODO check if horizontal or vertical fov
    nearCorner.x = nearCorner.y * aspect;

    corners[0] = inv_view * glm::vec4(nearCorner, -near, 1);
    corners[1] = inv_view * glm::vec4(-nearCorner, -near, 1);

    nearCorner.y *= -1;
    corners[2] = inv_view * glm::vec4(nearCorner, -near, 1);
    corners[3] = inv_view * glm::vec4(-nearCorner, -near, 1);

    glm::vec2 farCorner{0, 0,};
    farCorner.y = glm::tan(fovRad / 2) * -far; // TODO check if horizontal or vertical fov
    farCorner.x = farCorner.y * aspect;

    corners[4] = inv_view * glm::vec4(farCorner, -far, 1);
    corners[5] = inv_view * glm::vec4(-farCorner, -far, 1);

    farCorner.y *= -1;
    corners[6] = inv_view * glm::vec4(farCorner, -far, 1);
    corners[7] = inv_view * glm::vec4(-farCorner, -far, 1);

    for(auto& corner : corners) {
        corner /= corner.w;
        bMin = glm::min(corner.xyz(), bMin);
        bMax = glm::max(corner.xyz(), bMax);
    }
}

bool Frustum::test(const glm::vec3 &point) const {
    using namespace glm;
    const vec4 v = vec4(point, 1);
    float outside = 0;
    outside += step(dot(cp[LEFT_PLANE], v), 0.f) + step(dot(cp[RIGHT_PLANE], v), 0.f);
    outside += step(dot(cp[BOTTOM_PLANE], v), 0.f) + step(dot(cp[TOP_PLANE], v), 0.f);
    outside += step(dot(cp[NEAR_PLANE], v), 0.f) + step(dot(cp[FAR_PLANE], v), 0.f);

    return outside == 0;
}

bool Frustum::test(const glm::vec3 &bMin, const glm::vec3 &bMax) const {
    using namespace glm;

    auto corners = std::array<vec4, 8> {{
        vec4(bMin.x, bMin.y, bMin.z, 1), vec4(bMax.x, bMin.y, bMin.z, 1), vec4(bMin.x, bMax.y, bMin.z, 1),
        vec4(bMax.x, bMax.y, bMin.z, 1), vec4(bMin.x, bMin.y, bMax.z, 1), vec4(bMax.x, bMin.y, bMax.z, 1),
        vec4(bMin.x, bMax.y, bMax.z, 1), vec4(bMax.x, bMax.y, bMax.z, 1)
    }};

    for(int i = 0; i < 6; ++i) {
        float outside = 0;
        outside += step(dot( cp[i], corners[0] ) , 0.f );
        outside += step(dot( cp[i], corners[1] ) , 0.f );
        outside += step(dot( cp[i], corners[2] ) , 0.f );
        outside += step(dot( cp[i], corners[3] ) , 0.f );
        outside += step(dot( cp[i], corners[4] ) , 0.f );
        outside += step(dot( cp[i], corners[5] ) , 0.f );
        outside += step(dot( cp[i], corners[6] ) , 0.f );
        outside += step(dot( cp[i], corners[7] ) , 0.f );

        if (outside == 8) return false;
    }

    return true;
}

bool Frustum::test(const glm::vec3 &boxCenter, float scale) {
    using namespace glm;
    static auto corners = std::array<glm::vec4, 8> {{
        vec4( -0.5, -0.5, -0.5, 0.5 ), vec4(0.5, -0.5, -0.5, 0.5 ), vec4(0.5, -0.5, 0.5, 0.5 ), vec4(-0.5, -0.5, 0.5, 0.5 ),
        vec4( -0.5, 0.5, -0.5, 0.5 ), vec4(0.5, 0.5, -0.5, 0.5 ), vec4(0.5, 0.5, 0.5, 0.5 ), vec4(-0.5, 0.5, 0.5, 0.5 ),
    }};


    const auto bc = glm::vec4(boxCenter, 0.5);
    const auto s = glm::vec4(scale, scale, scale, 1);
    for(int i = 0; i < 6; ++i) {
        float outside = 0;
        outside += step(dot( cp[i], bc + corners[0] * s), 0.f);
        outside += step(dot( cp[i], bc + corners[1] * s), 0.f);
        outside += step(dot( cp[i], bc + corners[2] * s), 0.f);
        outside += step(dot( cp[i], bc + corners[3] * s), 0.f);
        outside += step(dot( cp[i], bc + corners[4] * s), 0.f);
        outside += step(dot( cp[i], bc + corners[5] * s), 0.f);
        outside += step(dot( cp[i], bc + corners[6] * s), 0.f);
        outside += step(dot( cp[i], bc + corners[7] * s), 0.f);

        if (outside == 8) return false;
    }

    return true;
}

void Frustum::extractFrustum(Frustum &frustum, const glm::mat4 M) {
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
}
