#include "AbstractCamera.hpp"
#include "random.h"
#include "xforms.h"
#include <gtest/gtest.h>
#include <vector>

#define NYI FAIL() << "Not yet implemented!";

static struct Clip{
    glm::vec3 min{-1, -1, 0};
    glm::vec3 max{1};
} clip;

struct FrustumTestFixture : public testing::Test {
protected:
    using Box = std::tuple<glm::vec3, glm::vec3>;

    void SetUp() override {
        view = glm::lookAt({0, 1, 10}, glm::vec3(0), {0, 1, 0});
        projection = vkn::perspective(glm::radians(60.f), 1.f, 1.f, 10.f);
        VP = projection * view;
    }

    static void pointsInFrustumCheck(const glm::mat4& M, const std::vector<glm::vec3>& points) {
        Frustum frustum{};
        AbstractCamera::extractFrustum(frustum, M);

        for(const auto& p : points) {
            ASSERT_TRUE(frustum.test(p));
        }
    }
    
    static void boxInFrustumCheck(const glm::mat4& M, const std::vector<Box>& boxs) {
        Frustum frustum{};
        AbstractCamera::extractFrustum(frustum, M);
        
        for(const auto& box : boxs) {
            const auto [bMin, bMax] = box;
            ASSERT_TRUE(frustum.test(bMin, bMax));
        }
    }

    static void boxInFrustumCheck(const glm::mat4& M, const std::vector<glm::vec3>& boxs) {
        Frustum frustum{};
        AbstractCamera::extractFrustum(frustum, M);

        for(const auto& box : boxs) {
            ASSERT_TRUE(frustum.test(box, 1));
        }
    }

    static void boxOutsideFrustumCheck(const glm::mat4& M, const std::vector<glm::vec3>& boxs) {
        Frustum frustum{};
        AbstractCamera::extractFrustum(frustum, M);

        for(const auto& box : boxs) {
            ASSERT_FALSE(frustum.test(box, 1));
        }
    }

    static void boxOutsideFrustumCheck(const glm::mat4& M, const std::vector<Box>& boxes) {
        Frustum frustum{};
        AbstractCamera::extractFrustum(frustum, M);

        for(const auto & box : boxes) {
            const auto [bMin, bMax] = box;
            ASSERT_FALSE(frustum.test(bMin, bMax));
        }
    }


    static void pointsOutsideFrustumCheck(const glm::mat4& M, const std::vector<glm::vec3>& points) {
        Frustum frustum{};
        AbstractCamera::extractFrustum(frustum, M);

        for(const auto& p : points) {
            ASSERT_FALSE(frustum.test(p));
        }
    }

    static std::vector<glm::vec3> generateRandomPoints(const glm::mat4& M, uint32_t N) {

        auto invM = glm::inverse(M);
        auto seed = glm::uvec3(1 << 20);
        std::vector<glm::vec3> points;
        for(auto i = 0u; i < N; ++i) {
            auto p = randomVec3(clip.min + 0.01f, clip.max - 0.01f, seed + i);
            auto pp = invM * glm::vec4(p, 1);
            pp /= pp.w;
            points.push_back(pp.xyz());
        }

        return points;
    }

    static std::vector<Box> generateRandomBox_InsideClipSpace(const glm::mat4& M, uint32_t N, float smin = 1, float smax = 1) {
        const auto d = glm::vec3(1, 1, -1);
        const auto points = generateRandomPoints(M, N);
        auto scale = rng(smin, smax, 1 << 20);

        std::vector<Box> boxs;

        for(auto p : points) {
//            auto bMin = p;
//            auto bMax = p + d * scale();
//            boxs.emplace_back(bMin, bMax);
            auto box = boxAround(p, scale());
            boxs.emplace_back(box);
        }

        return boxs;
    }
    static std::vector<Box> generateRandomBoxs_outsideClipSpace(const glm::mat4& M, uint32_t N, float smin = 1, float smax = 1) {
        const auto points = generateRandomPointsOutSide(M, N, (smax + 0.1f) * 0.5f, smax * 5);
        auto scale = rng(smin, smax, 1 << 20);

        std::vector<Box> boxs;

        for(const auto& p : points) {
            auto box = boxAround(p, scale());
            boxs.emplace_back(box);
        }

        return boxs;
    }

    static Box boxAround(const glm::vec3& point, float scale) {
        static std::array<glm::vec3, 8> corners {{
                {1, 1, 1}, {1, -1, 1}, {-1, -1, 1}, {-1, 1, 1},
                {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}, {-1, 1, -1},
        }};

        auto hScale = scale * 0.5f;
        auto bMin = glm::vec3(MAX_FLOAT);
        auto bMax = glm::vec3(MIN_FLOAT);

        for(const auto& c : corners) {
            const auto d = glm::normalize(c);
            const auto p = point + d * hScale;
            bMin = glm::min(bMin, p);
            bMax = glm::max(bMax, p);
        }

        return std::make_tuple(bMin, bMax);
    }

    static std::vector<glm::vec3> generateRandomPointsOutSide(const glm::mat4& M, uint32_t N, float margin = 0.1f, float dist = 2.f) {
        const auto clipMin = clip.min - margin;
        const auto clipMax = clip.max + margin;

        const auto invM = glm::inverse(M);
        auto seed = glm::uvec3(1 << 20);

        auto lx = rng(clipMin.x - dist, clipMin.x, seed.x);
        auto rx = rng(clipMax.x, clipMax.x + dist, seed.x);
        auto ly = rng(clipMin.y - dist, clipMin.y, seed.y);
        auto ry = rng(clipMax.y, clipMax.y + dist, seed.y);
        auto lz = rng(clipMin.z - dist, clipMin.z, seed.z);
        auto rz = rng(clipMax.z, clipMax.z + dist, seed.z);

        auto x = [&]{ return coinFlip(seed.x) ? lx() : rx(); };
        auto y = [&]{ return coinFlip(seed.y) ? ly() : ry(); };
        auto z = [&]{ return coinFlip(seed.z) ? lz() : rz(); };

        auto pointOutsideClipSpace = [&] {
            return glm::vec3(x(), y(), z());
        };

        std::vector<glm::vec3> points;
        for(auto i = 0u; i < N; ++i) {
            auto p = pointOutsideClipSpace();
            auto pp = invM * glm::vec4(p, 1);
            pp /= pp.w;
            points.push_back(pp.xyz());
        }

        return points;
    }

    glm::mat4 projection{1};
    glm::mat4 view{1};
    glm::mat4 VP{1};
};

TEST_F(FrustumTestFixture, pointInProjectionFrustum) {
    auto points = generateRandomPoints(projection, 100);

    pointsInFrustumCheck(projection, points);
}

TEST_F(FrustumTestFixture, pointInViewProjectionFrustum) {
    auto points = generateRandomPoints(VP, 100);

    pointsInFrustumCheck(VP, points);

}

TEST_F(FrustumTestFixture, pointsOutsizeProjectionFrustum) {
    auto points = generateRandomPointsOutSide(projection, 100);
    points.emplace_back(glm::vec4(-0.19485605f, 0.19485605f, -0.6962495f, 1.f));

    pointsOutsideFrustumCheck(projection, points);
}

TEST_F(FrustumTestFixture, pointsOutsizeViewProjectionFrustum) {
    auto points = generateRandomPointsOutSide(VP, 100);

    pointsOutsideFrustumCheck(VP, points);
}

TEST_F(FrustumTestFixture, BoxinProjectionFrustum) {
    auto boxes = generateRandomBox_InsideClipSpace(projection, 100, 0.5, 2.5);

    boxInFrustumCheck(projection, boxes);
}

TEST_F(FrustumTestFixture, BoxInViewProjectionFrustum) {
    auto boxes = generateRandomBox_InsideClipSpace(VP, 100, 0.5, 2.5);

    boxInFrustumCheck(VP, boxes);
}

TEST_F(FrustumTestFixture, BoxOutsideProjectionFrustum) {
    const auto boxes = generateRandomBoxs_outsideClipSpace(projection, 100, 0.5, 2.5);

    boxOutsideFrustumCheck(projection, boxes);
}

TEST_F(FrustumTestFixture, BoxOutsideViewProjectionFrustum) {
    const auto boxes = generateRandomBoxs_outsideClipSpace(VP, 100, 0.5, 2.5);

    boxOutsideFrustumCheck(VP, boxes);
}

TEST_F(FrustumTestFixture, BoxInProjectionFrustumGivenCenter) {
    const auto boxes = generateRandomPoints(projection, 100);

    boxInFrustumCheck(projection, boxes);
}

TEST_F(FrustumTestFixture, BoxInViewProjectionFrustumGivenCenter) {
    const auto boxes = generateRandomPoints(VP, 100);

    boxInFrustumCheck(VP, boxes);
}

TEST_F(FrustumTestFixture, BoxOutsideProjectionFrustumGivenCenter) {
    const auto boxes = generateRandomPointsOutSide(projection, 100, 1.0, 5.0);
    boxOutsideFrustumCheck(projection, boxes);
}

TEST_F(FrustumTestFixture, BoxOutsideViewProjectionFrustumGivenCenter) {
    const auto boxes = generateRandomPointsOutSide(VP, 100, 1.0, 5.0);
    boxOutsideFrustumCheck(VP, boxes);
}