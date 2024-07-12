#include "primitives.h"

static constexpr auto PI = glm::pi<float>();

struct Surface {
    glm::vec3 position{};
    glm::vec3 normal{0, 1, 0};
    glm::vec3 tangent{1, 0, 0};
    glm::vec3 bitangent{0, 0, 1};
};

Vertices primitives::cube(const glm::vec4& color){
    Vertices mesh;

    mesh.vertices = {

            // FRONT FACE
            {{-1.0, -1.0, 1.0, 1}, color, {0.0f, 0.0f, 1.0f}, {1, 0, 0}, {0, 1, 0},  {0, 0}},
            {{1.0, -1.0, 1.0, 1}, color,  {0.0f, 0.0f, 1.0f}, {1, 0, 0}, {0, 1, 0}, {1, 0}},
            {{1.0, 1.0, 1.0, 1}, color,  {0.0f, 0.0f, 1.0f}, {1, 0, 0}, {0, 1, 0}, {1, 1}},
            {{-1.0, 1.0, 1.0, 1}, color,  {0.0f, 0.0f, 1.0f}, {1, 0, 0}, {0, 1, 0}, {0, 1}},

            // RIGHT FACE
            {{1.0, -1.0, 1.0, 1}, color,  {1.0f, 0.0f, 0.0f}, {0, 0, -1}, {0, 1, 0}, {0, 0}},
            {{1.0, -1.0, -1.0, 1}, color,  {1.0f, 0.0f, 0.0f}, {0, 0, -1}, {0, 1, 0}, {1, 0}},
            {{1.0, 1.0, -1.0, 1}, color,  {1.0f, 0.0f, 0.0f}, {0, 0, -1}, {0, 1, 0}, {1, 1}},
            {{1.0, 1.0, 1.0, 1}, color,  {1.0f, 0.0f, 0.0f}, {0, 0, -1}, {0, 1, 0}, {0, 1}},

            // BACK FACE
            {{-1.0, -1.0, -1.0, 1}, color, {0.0f, 0.0f, -1.0f}, {-1, 0, 0}, {0, 1, 0}, {1, 0}},
            {{-1.0, 1.0, -1.0, 1}, color, {0.0f, 0.0f, -1.0f}, {-1, 0, 0}, {0, 1, 0}, {1, 1}},
            {{1.0, 1.0, -1.0, 1}, color, {0.0f, 0.0f, -1.0f}, {-1, 0, 0}, {0, 1, 0}, {0, 1}},
            {{1.0, -1.0, -1.0, 1}, color, {0.0f, 0.0f, -1.0f}, {-1, 0, 0}, {0, 1, 0}, {0, 0}},

            // LEFT FACE
            {{-1.0, -1.0, 1.0, 1},  color, {-1.0f, 0.0f, 0.0f}, {0, 0, 1}, {0, 1, 0}, {1, 0}},
            {{-1.0, 1.0, 1.0, 1}, color, {-1.0f, 0.0f, 0.0f}, {0, 0, 1}, {0, 1, 0},  {1, 1}},
            {{-1.0, 1.0, -1.0, 1}, color, {-1.0f, 0.0f, 0.0f}, {0, 0, 1}, {0, 1, 0},  {0, 1}},
            {{-1.0, -1.0, -1.0, 1}, color, {-1.0f, 0.0f, 0.0f}, {0, 0, 1}, {0, 1, 0},  {0, 0}},

            // BOTTOM FACE
            {{-1.0, -1.0, 1.0, 1}, color, {0.0f, -1.0f, 0.0f}, {1, 0, 0}, {0, 0, 1},  {0, 1}},
            {{-1.0, -1.0, -1.0, 1}, color, {0.0f, -1.0f, 0.0f}, {1, 0, 0}, {0, 0, 1},  {0, 0}},
            {{1.0, -1.0, -1.0, 1}, color, {0.0f, -1.0f, 0.0f}, {1, 0, 0}, {0, 0, 1},  {1, 0}},
            {{1.0, -1.0, 1.0, 1}, color, {0.0f, -1.0f, 0.0f}, {1, 0, 0}, {0, 0, 1},  {1, 1}},

            // TOP FACE
            {{-1.0, 1.0, 1.0, 1}, color,  {0.0f, 1.0f, 0.0f}, {1, 0, 0}, {0, 0, -1},  {0, 0}},
            {{1.0, 1.0, 1.0, 1}, color,  {0.0f, 1.0f, 0.0f}, {1, 0, 0}, {0, 0, -1},  {1, 0}},
            {{1.0, 1.0, -1.0, 1}, color,  {0.0f, 1.0f, 0.0f}, {1, 0, 0}, {0, 0, -1},  {1, 1}},
            {{-1.0, 1.0, -1.0, 1}, color,  {0.0f, 1.0f, 0.0f}, {1, 0, 0}, {0, 0, -1},  {0, 1}},
    };

    mesh.indices = {
            0,1,2,0,2,3,
            4,5,6,4,6,7,
            8,9,10,8,10,11,
            12,13,14,12,14,15,
            16,17,18,16,18,19,
            20,21,22,20,22,23
    };
    return mesh;
}

Vertices primitives::cubeOutline(const glm::vec4& color) {
    Vertices vertices{
        .vertices = {
// FRONT
                { .position = glm::vec4(-1, -1, -1, 1), .color = color},
                { .position = glm::vec4(-1, 1, -1, 1), .color = color},

                { .position = glm::vec4(-1, 1, -1, 1), .color = color},
                { .position = glm::vec4(1, 1, -1, 1), .color = color},

                { .position = glm::vec4(1, 1, -1, 1), .color = color},
                { .position = glm::vec4(1, -1, -1, 1), .color = color},

                { .position = glm::vec4(1, -1, -1, 1), .color = color},
                { .position = glm::vec4(-1, -1, -1, 1), .color = color},

// BACK
                { .position = glm::vec4(-1, -1, 1, 1), .color = color},
                { .position = glm::vec4(-1, 1, 1, 1), .color = color},

                { .position = glm::vec4(-1, 1, 1, 1), .color = color},
                { .position = glm::vec4(1, 1, 1, 1), .color = color},

                { .position = glm::vec4(1, 1, 1, 1), .color = color},
                { .position = glm::vec4(1, -1, 1, 1), .color = color},

                { .position = glm::vec4(1, -1, 1, 1), .color = color},
                { .position = glm::vec4(-1, -1, 1, 1), .color = color},


// SIDES
                { .position = glm::vec4(-1, -1, -1, 1), .color = color},
                { .position = glm::vec4(-1, -1, 1, 1), .color = color},

                { .position = glm::vec4(-1, 1, -1, 1), .color = color},
                { .position = glm::vec4(-1, 1, 1, 1), .color = color},

                { .position = glm::vec4(1, 1, -1, 1), .color = color},
                { .position = glm::vec4(1, 1, 1, 1), .color = color},

                { .position = glm::vec4(1, -1, -1, 1), .color = color},
                { .position = glm::vec4(1, -1, 1, 1), .color = color},
        }
    };

    return vertices;

}

Vertices primitives::sphere(int rows, int columns, float radius, glm::mat4 xform, const glm::vec4 &color, VkPrimitiveTopology topology) {
    const auto p = columns;
    const auto q = rows;
    const auto r = radius;

    auto f = [&](float i, float j) {
        float u = 2 * i / p * PI;
        float v = j / q * PI;

        float nx = std::cos(u) * std::sin(v);
        float x = r * nx;

        float ny = std::cos(v);
        float y = r * ny;

        float nz = std::sin(u) * std::sin(v);
        float z = r * nz;

        float tx = -r * std::sin(u) * std::sin(v);
        float ty = 0;
        float tz = r * std::cos(u) * std::sin(v);

        float bx = r * std::cos(u) * std::cos(v);
        float by = -r * std::sin(v);
        float bz = r * std::sin(u) * std::cos(v);

       return Surface{ {x, y, z}, {nx, ny, nz}, {tx, ty, tz}, {bx, by, bz} };
    };

    return generateSurface(p, q, f, color, xform, topology);
}

Vertices primitives::hemisphere(int rows, int columns, float radius, const glm::vec4 &color, VkPrimitiveTopology topology) {
    auto p = columns;
    auto q = rows;

    auto f = [&](float i, float j) {
        float u = 2 * i / p * PI;
        float v = j / q * PI/2.0f;

        float nx = std::cos(u) * std::sin(v);
        float x = radius * nx;

        float ny = std::cos(v);
        float y = radius * ny;

        float nz = std::sin(u) * std::sin(v);
        float z = radius * nz;

        return Surface{ glm::vec3(x, y, z), glm::vec3(nx, ny, nz) };
    };

    return generateSurface(p, q, f, color, glm::mat4{1}, topology);
}

Vertices primitives::cone(int rows, int columns, float radius, float height, const glm::vec4 &color, VkPrimitiveTopology topology) {
    const auto p = columns;
    const auto q = rows;
    const auto h = height;

    auto f = [&](float i, float j) {
        float u = 2 * i / p * PI;
        float v =  j/q * h;

        float nx = std::cos(u);
        float x = radius * v * std::cos(u);

        float ny =  std::sin(u);
        float y = radius  * v * std::sin(u);

        float nz = 0;
        float z =  v - h * 0.5f;

        return Surface{ glm::vec3(x, y, z), glm::vec3(nx, ny, nz) };
    };

    return generateSurface(p, q, f, color, glm::mat4{1}, topology);
}

Vertices primitives::cylinder(int rows, int columns, float radius, float height, const glm::vec4 &color, VkPrimitiveTopology topology) {
    const auto p = columns;
    const auto q = rows;
    const auto h = height;

    auto f = [&](float i, float j) {
        float u = (-1.f + 2.f * i/p) * PI;
        float v =   j/q * h;
        float nx = std::sin(u);
        float x = radius *  height * std::sin(u);

        float ny = 0;
        float y = v - h * 0.5f;

        float nz = std::cos(u);
        float z = radius * height * std::cos(u);

        return Surface{ glm::vec3(x, y, z), glm::vec3(nx, ny, nz) };
    };

    return generateSurface(p, q, f, color, glm::mat4{1}, topology);
}

Vertices primitives::torus(int rows, int columns, float innerRadius, float outerRadius,  glm::mat4 xform, const glm::vec4 &color, VkPrimitiveTopology topology) {
    auto p = columns;
    auto q = rows;
    auto R = innerRadius;
    auto r = outerRadius;

    auto f = [&](float i, float j) {
        float u = (-1.f + 2.f * i/p) * PI;
        float v =  (-1.f + 2.f * j/q) * PI;

        float x = (R + r * std::cos(v)) * std::cos(u);
        float nx = std::cos(v) * std::cos(u);

        float y = (R + r * std::cos(v)) * std::sin(u);
        float ny = std::cos(v) * sin(u);

        float z = r * std::sin(v);
        float nz =  std::sin(v);

        return Surface{ glm::vec3(x, y, z), glm::vec3(nx, ny, nz) };
    };


    return generateSurface(p, q, f, color, xform, topology);
}

Vertices primitives::plane(int rows, int columns, float width, float height, const glm::mat4& xform, const glm::vec4 &color, VkPrimitiveTopology topology) {

    const auto p = columns;
    const auto q = rows;
    const auto halfWidth = width * 0.5f;
    const auto halfHeight = height * 0.5f;

    auto f = [&](float i, float j) {
        float u = i / p * width - halfWidth;
        float v = j / q * height - halfHeight;

        float x = u;
        float nx = 0;

        float y = v;
        float ny = 0;

        float z = 0;
        float nz = 1;


        return Surface{ glm::vec3(x, y, z), glm::vec3(nx, ny, nz) };
    };

    auto vertices = generateSurface(p, q, f, color, xform, topology);

    glm::vec3 N{0.0f, 0.0f, 1.0f};
    glm::vec3 T{1, 0, 0};
    glm::vec3 B{0, 1, 0};

    N = glm::inverseTranspose(glm::mat3(xform)) * N;
    T = glm::inverseTranspose(glm::mat3(xform)) * T;
    B = glm::inverseTranspose(glm::mat3(xform)) * B;

    for(auto& vertex : vertices.vertices) {
        vertex.normal = N;
        vertex.tangent = T;
        vertex.bitangent = B;
    }

    return vertices;
}



template<typename SurfaceFunction>
Vertices primitives::generateSurface(int p, int q, SurfaceFunction &&f, const glm::vec4 &color, const glm::mat4& xform, VkPrimitiveTopology topology) {
    Vertices vertices;
    vertices.topology = topology;
    auto nXform = glm::inverseTranspose(glm::mat3(xform));
    for (int j = 0; j <= q; j++) {
        for (int i = 0; i <= p; i++) {
            auto [position, normal, tangent, bitangent] = f(i, j);
            Vertex vertex{};
            vertex.position = xform * glm::vec4(position, 1.0);
            vertex.normal =  nXform * normal;
            vertex.tangent = nXform * tangent;
            vertex.bitangent = nXform * bitangent;
            vertex.color = color;
            vertex.uv = {float(i) / float(p), float(j) / float(q)};
            vertices.vertices.push_back(vertex);
        }
    }

    if(topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST){
        for (int j = 0; j < q; j++) {
            for (int i = 0; i < p; i++) {
                vertices.indices.push_back((j + 1) * (p + 1) + i);
                vertices.indices.push_back(j * (p + 1) + i);
                vertices.indices.push_back((j + 1) * (p + 1) + i + 1);

                vertices.indices.push_back((j + 1) * (p + 1) + i + 1);
                vertices.indices.push_back(j * (p + 1) + i);
                vertices.indices.push_back(j * (p + 1) + i + 1);

            }
        }
    }else {
        for (int j = 0; j < q; j++) {
            for (int i = 0; i <= p; i++) {
                vertices.indices.push_back((j + 1) * (p + 1) + i);
                vertices.indices.push_back(j * (p + 1) + i);
            }
            vertices.indices.push_back(RESTART_PRIMITIVE);
        }
    }

    return vertices;
}

Vertices primitives::triangleStripToTriangleList(const Vertices &vertices) {
    if(vertices.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) return vertices;

    Vertices newVertices{};
    newVertices.vertices = vertices.vertices;
    newVertices.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    if(vertices.indices.empty()){
        assert(vertices.vertices.size() > 2);
        int numFaces = vertices.vertices.size() - 2;

        int i = 0;
        for(; i < numFaces; i++){
            int v0, v1, v2;
            if(i%2 == 0){
                v0 = i;
                v1 = i+1;
                v2 = i+2;
            }else{
                v0 = i+1;
                v1 = i;
                v2 = i+2;
            }
            newVertices.indices.push_back(v0);
            newVertices.indices.push_back(v1);
            newVertices.indices.push_back(v2);
        }
        return newVertices;
    }
    assert(vertices.indices.size() > 2);
    int numFaces = vertices.indices.size() - 2; // we are assuming there are N faces defined by N+2 vertices

    for(auto i = 0, restarts = 0; i < numFaces; i++){
        int v0, v1, v2;
        if((i+restarts)%2 == 0){
            v0 = i;
            v1 = i+1;
            v2 = i+2;
        }else{
            v0 = i+1;
            v1 = i;
            v2 = i+2;
        }
        auto i0 = vertices.indices[v0];
        auto i1 = vertices.indices[v1];
        auto i2 = vertices.indices[v2];

        if(i0 == RESTART_PRIMITIVE || i1 == RESTART_PRIMITIVE || i2 == RESTART_PRIMITIVE){
            restarts++;
            continue;
        }

        newVertices.indices.push_back(i0);
        newVertices.indices.push_back(i1);
        newVertices.indices.push_back(i2);
    }

    return newVertices;
}

Vertices primitives::calculateTangents(Vertices &vertices, bool smooth /* TODO implement smooth tangents */) {
    auto& indices = vertices.indices;

    for(int i = 0; i < indices.size(); i+= 3){
        auto& v0 = vertices.vertices[indices[i]];
        auto& v1 = vertices.vertices[indices[i+1]];
        auto& v2 = vertices.vertices[indices[i+2]];

        auto e1 = v1.position.xyz() - v0.position.xyz();
        auto e2 = v2.position.xyz() - v0.position.xyz();

        auto du1 = v1.uv.x - v0.uv.x;
        auto dv1 = v1.uv.y - v0.uv.y;
        auto du2 = v2.uv.x - v0.uv.x;
        auto dv2 = v2.uv.y - v0.uv.y;

        auto d = 1.f/(du1 * dv2 - dv1 * du2);

        glm::vec3 tn{0};
        tn.x = d * (dv2 * e1.x - dv1 * e2.x);
        tn.y = d * (dv2 * e1.y - dv1 * e2.y);
        tn.z = d * (dv2 * e1.z - dv1 * e2.z);

        glm::vec3 bn{0};
        bn.x = d * (du1 * e2.x - du2 * e1.x);
        bn.y = d * (du1 * e2.y - du2 * e1.y);
        bn.z = d * (du1 * e2.z - du2 * e1.z);

        v0.tangent = normalize(tn);
        v1.tangent = normalize(tn);
        v2.tangent = normalize(tn);

        v0.bitangent = normalize(bn);
        v1.bitangent = normalize(bn);
        v2.bitangent = normalize(bn);
    }

    return vertices;
}

std::vector<Vertices> primitives::cornellBox() {
    auto white = glm::vec4{0.73, 0.71, 0.68, 1};
    auto red = glm::vec4{0.63, 0.064, 0.005, 1};
    auto green = glm::vec4{0.14, 0.45, 0.09, 1};
    auto w = 55.f;

    glm::mat4 xform = glm::translate(glm::mat4(1), {0, 0, -w * 0.5f});
    auto backWall = primitives::plane(1, 1, w, w, xform, white, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);


    xform = glm::translate(glm::mat4(1), {0, -w * 0.5f, 0});
    xform = glm::rotate(xform, -glm::half_pi<float>(), {1, 0, 0});
    auto floor = primitives::plane(1, 1, w, w, xform, white, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    xform = glm::translate(glm::mat4(1), {0, w * 0.5f, 0});
    xform = glm::rotate(xform, glm::half_pi<float>(), {1, 0, 0});
    auto ceiling = primitives::plane(1, 1, w, w, xform, white, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    xform = glm::translate(glm::mat4(1), {-w * 0.5f, 0, 0});
    xform = glm::rotate(xform, glm::half_pi<float>(), {0, 1, 0});
    auto rightWall = primitives::plane(1, 1, w, w, xform, red, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    xform = glm::translate(glm::mat4(1), {w * 0.5f, 0, 0});
    xform = glm::rotate(xform, -glm::half_pi<float>(), {0, 1, 0});
    auto leftWall = primitives::plane(1, 1, w, w, xform, green, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    xform = glm::translate(glm::mat4(1), {0, w * 0.5f - 0.1, 0});
    xform = glm::rotate(xform, glm::half_pi<float>(), {1, 0, 0});
    auto light = primitives::plane(1, 1, 13, 10.5, xform, glm::vec4(0), VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);


    xform = glm::translate(glm::mat4(1), glm::vec3(10, (16.6 - w) * 0.5, 12));
    xform = glm::rotate(xform, glm::radians(-18.f), {0, 1, 0});
    xform = glm::scale(xform, glm::vec3(16.5));
    auto shortBox = primitives::cube(white);

    auto nxForm = glm::inverseTranspose(glm::mat3(xform));
    for(auto& vertex : shortBox.vertices){
        vertex.position = xform * vertex.position;
        vertex.normal = normalize(nxForm * vertex.normal);
        vertex.tangent = normalize(nxForm * vertex.tangent);
        vertex.bitangent = normalize(nxForm * vertex.bitangent);
    }

//    xform = glm::translate(glm::mat4(1), glm::vec3(-26.5, (33 - w) * 0.5, -29.5));
    xform = glm::translate(glm::mat4(1), glm::vec3(-10.5, (33.1 - w) * 0.5, -5));
    xform = glm::rotate(xform, glm::radians(15.f), {0, 1, 0});
    xform = glm::scale(xform, glm::vec3(16.5, 33, 16.5));
    auto tallBox = primitives::cube(white);

    nxForm = glm::inverseTranspose(glm::mat3(xform));
    for(auto& vertex : tallBox.vertices){
        vertex.position = xform * vertex.position;
        vertex.normal = normalize(nxForm * vertex.normal);
        vertex.tangent = normalize(nxForm * vertex.tangent);
        vertex.bitangent = normalize(nxForm * vertex.bitangent);
    }

    return {
        light, ceiling, rightWall, floor, leftWall, tallBox, shortBox, backWall
    };
}