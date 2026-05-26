#pragma once

#include "VulkanRAII.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <array>

template<typename Scalar>
struct CameraT {
    using Mat4 = glm::mat<4, 4, Scalar, glm::defaultp>;

    Mat4 model = Mat4(1);
    Mat4 view = Mat4(1);
    Mat4 proj = Mat4(1);

    static constexpr VkPushConstantRange pushConstant(VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) {
        return {stageFlags, 0, sizeof(CameraT)};
    }
};

enum PlaneType : int { LEFT_PLANE = 0, RIGHT_PLANE, BOTTOM_PLANE, TOP_PLANE, NEAR_PLANE, FAR_PLANE};

template<typename Scalar>
using ClipPlaneT = glm::vec<4, Scalar, glm::defaultp>;

template<typename Scalar>
struct FrustumT {
    using Vec3 = glm::vec<3, Scalar, glm::defaultp>;
    using Mat4 = glm::mat<4, 4, Scalar, glm::defaultp>;
    using ClipPlane = ClipPlaneT<Scalar>;

    std::array<ClipPlane, 6> cp;

    bool test(const Vec3& point) const;

    bool test(const Vec3& boxMin, const Vec3& boxMax) const;

    bool test(const Vec3& boxCenter, Scalar scale);

    static void extractFrustum(FrustumT& frustum, Mat4 M);
};

template<typename Scalar>
class AbstractCameraT {
public:
    using Vec3 = glm::vec<3, Scalar, glm::defaultp>;
    using Mat4 = glm::mat<4, 4, Scalar, glm::defaultp>;
    using Quat = glm::qua<Scalar, glm::defaultp>;
    using Camera = CameraT<Scalar>;
    using Frustum = FrustumT<Scalar>;

    virtual ~AbstractCameraT() = default;

    virtual void update(float time) = 0;

    virtual void processInput() = 0;

    virtual void lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) = 0;

    virtual void perspective(Scalar fovx, Scalar aspect, Scalar znear, Scalar zfar) = 0;

    virtual void perspective(Scalar aspect) = 0;

    virtual void rotateSmoothly(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) = 0;

    virtual void rotate(Scalar headingDegrees, Scalar pitchDegrees, Scalar rollDegrees) = 0;

    virtual void move(Scalar dx, Scalar dy, Scalar dz) = 0;

    virtual void move(const Vec3& direction, const Vec3& amount) = 0;

    virtual void position(const Vec3& pos) = 0;

    [[nodiscard]]
    virtual const Vec3& position() const = 0;

    [[nodiscard]]
    virtual const Vec3& velocity() const = 0;

    [[nodiscard]]
    virtual const Vec3& acceleration() const = 0;

    virtual Scalar near() const = 0;

    virtual Scalar far() const = 0;

    virtual void fieldOfView(Scalar value) = 0;

    virtual void updatePosition(const Vec3& direction, Scalar elapsedTimeSec) = 0;

    virtual void undoRoll() = 0;

    virtual void zoom(Scalar zoom, Scalar minZoom, Scalar maxZoom) = 0;

    virtual void onResize(int width, int height) = 0;

    virtual void setModel(const Mat4& model) = 0;

    virtual void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) const = 0;

    virtual void push(VkCommandBuffer commandBuffer, VulkanPipelineLayout layout, const Mat4& model, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT) = 0;

    [[nodiscard]]
    virtual const Quat& getOrientation() const = 0;

    [[nodiscard]]
    virtual const Camera& cam() const = 0;

    virtual const Camera& previousCamera() const = 0;

    virtual void newFrame() = 0;

    [[nodiscard]]
    virtual bool moved() const = 0;

    virtual void jitter(Scalar jx, Scalar jy) = 0;

    virtual void extract(Frustum& frustum) const = 0;

    virtual void extractAABB(Vec3& bMin, Vec3& bMax) const = 0;
};

using Camera = CameraT<float>;
using DoubleCamera = CameraT<double>;

using ClipPlane = ClipPlaneT<float>;
using DoubleClipPlane = ClipPlaneT<double>;

using Frustum = FrustumT<float>;
using DoubleFrustum = FrustumT<double>;

using AbstractCamera = AbstractCameraT<float>;
using DoubleAbstractCamera = AbstractCameraT<double>;
