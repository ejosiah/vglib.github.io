#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "BoxSurfaceFixture.hpp"
#include <vector>
#include <glm/gtc/epsilon.hpp>
#include <fmt/format.h>
#include "glm_format.h"

template<glm::length_t L, typename T, glm::qualifier Q>
bool vectorEquals(glm::vec<L, T, Q> v0, glm::vec<L, T, Q> v1, float eps = 1e-4){
    return glm::all(glm::epsilonEqual(v0, v1, eps));
}

TEST_F(BoxSurfaceFixture, closestPointWhenPointInsideBox){
    auto [surface, planes] = createBoxSurface(-0.5, 0.5);

    vec3 mid = midPoint(surface.bounds);
    for(auto& p : planes){

        vec3 expectedPointOnSurface = closestPoint(p, mid);
        vec3 point = expectedPointOnSurface - 0.2f * p.normal;

        float closestDist;
        vec3 closestNormal;
        vec3 actualPointOnSurface = closestPoint(surface, point, closestNormal, closestDist);
        ASSERT_TRUE(vectorEquals(expectedPointOnSurface, actualPointOnSurface));
        ASSERT_TRUE(vectorEquals(p.normal, closestNormal));
        ASSERT_FLOAT_EQ(0.2f, closestDist);
    }
}

TEST_F(BoxSurfaceFixture, closestOnSurfaceWhenPointOutsideBox){
    auto [surface, planes] = createBoxSurface(-0.5, 0.5);

    vec3 mid = midPoint(surface.bounds);
    for(auto& p : planes){

        vec3 expectedPointOnSurface = closestPoint(p, mid);
        vec3 point = expectedPointOnSurface + 0.2f * p.normal;

        float closestDist;
        vec3 closestNormal;
        vec3 actualPointOnSurface = closestPoint(surface, point, closestNormal, closestDist);
        ASSERT_TRUE(vectorEquals(expectedPointOnSurface, actualPointOnSurface));
        ASSERT_TRUE(vectorEquals(p.normal, closestNormal));
        ASSERT_FLOAT_EQ(0.2f, closestDist);
    }
}

TEST_F(BoxSurfaceFixture, returnFlipedNormalOnClosestPointIfEnabled){
    auto [surface, planes] = createBoxSurface(-0.5, 0.5);
    surface.normalFlipped = true;
    vec3 mid = midPoint(surface.bounds);
    for(auto& p : planes){

        vec3 expectedPointOnSurface = closestPoint(p, mid);
        vec3 point = expectedPointOnSurface + 0.2f * p.normal;

        float closestDist;
        vec3 closestNormal;
        vec3 actualPointOnSurface = closestPoint(surface, point, closestNormal, closestDist);
        ASSERT_TRUE(vectorEquals(expectedPointOnSurface, actualPointOnSurface));
        ASSERT_TRUE(vectorEquals(-p.normal, closestNormal));
        ASSERT_FLOAT_EQ(0.2f, closestDist);
    }
}

TEST_F(BoxSurfaceFixture, returnFalseIfObjectIsNotPenetratingThroughSurface){
    auto [surface, _] = createBoxSurface({-0.5, 0, -0.5}, {0.5, 2, 0.5 });
    surface.normalFlipped = true;
    float radius = 0.02;
    glm::vec3 point{-0.044, 0.916, -0.084};

    vec3 normal;
    vec3 surfacePoint;
    ASSERT_FALSE(isPenetrating(surface, point, radius, normal, surfacePoint)) << "point should not penetrate through surface";
}