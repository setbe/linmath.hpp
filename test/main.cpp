#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#define LINMATH_FREESTANDING
#include "../linmath.hpp"

#ifdef LINMATH_FREESTANDING
// Only allowed for handcrafted functions (e.g. cosf, sinf, sqrtf, etc.)
LINMATH_CONSTEXPR_VAR float epsilon = 1e-2f;
#else
// Standard accuracy with `#include math.h`
LINMATH_CONSTEXPR_VAR float epsilon = 1e-6f;
#endif

using namespace lmath;

TEST_CASE("Vector operations", "[vec]") {
    vec3 a{ 1.f, 2.f, 3.f };
    vec3 b{ 4.f, 5.f, 6.f };

    SECTION("Addition") {
        vec3 r = a + b;
        REQUIRE(r.x == Approx(5.f));
        REQUIRE(r.y == Approx(7.f));
        REQUIRE(r.z == Approx(9.f));
    }

    SECTION("Subtraction") {
        vec3 r = a - b;
        REQUIRE(r.x == Approx(-3.f));
        REQUIRE(r.y == Approx(-3.f));
        REQUIRE(r.z == Approx(-3.f));
    }

    SECTION("Dot product") {
        float dot = vec3::dot(a, b);
        REQUIRE(dot == Approx(32.f)); // 1*4 + 2*5 + 3*6
    }

    SECTION("Cross product") {
        vec3 r = a.cross(b);
        REQUIRE(r.x == Approx(-3.f));
        REQUIRE(r.y == Approx(6.f));
        REQUIRE(r.z == Approx(-3.f));
    }

    SECTION("Length and normalization") {
        vec3 v{ 3.f, 0.f, 4.f };
        REQUIRE(v.length() == Approx(5.f).margin(epsilon));
        vec3 norm = v.normalized();
        REQUIRE(norm.length() == Approx(1.f).margin(epsilon));
    }

    SECTION("Reflect") {
        vec3 incident{ 1.f, -1.f, 0.f };
        vec3 normal{ 0.f, 1.f, 0.f };
        vec3 r = incident.reflect(normal);
        REQUIRE(r.x == Approx(1.f));
        REQUIRE(r.y == Approx(1.f));
        REQUIRE(r.z == Approx(0.f));
    }
} // TEST_CASE("Vector operations", "[vec]")

TEST_CASE("Matrix operations", "[mat4f]") {
    mat4f identity = mat4f::identity();
    vec4 v{ 1.f, 2.f, 3.f, 1.f };

    SECTION("Identity multiplication") {
        vec4 r = identity * v;
        REQUIRE(r.x == Approx(v.x));
        REQUIRE(r.y == Approx(v.y));
        REQUIRE(r.z == Approx(v.z));
        REQUIRE(r.w == Approx(v.w));
    }

    SECTION("Translation") {
        mat4f trans = mat4f::translate(10.f, 0.f, 0.f);
        vec4 r = trans * v;
        REQUIRE(r.x == Approx(11.f));
        REQUIRE(r.y == Approx(2.f));
        REQUIRE(r.z == Approx(3.f));
        REQUIRE(r.w == Approx(1.f));
    }

    SECTION("Rotation around X") {
        mat4f mat = mat4f::identity();
        mat4f rot_x = mat.rotate_x(lmath::PI_HALF);
        vec4 v0{ 0.f, 1.f, 0.f, 1.f };
        vec4 r = rot_x * v0;
        REQUIRE(r.x == Approx(0.f));
        REQUIRE(r.y == Approx(0.f).margin(1e-6f));
        REQUIRE(r.z == Approx(1.f).margin(epsilon));
        REQUIRE(r.w == Approx(1.f));
    }

    SECTION("Rotation around Y") {
        mat4f mat = mat4f::identity();
        mat4f rot_y = mat.rotate_y(lmath::PI_HALF);
        vec4 v0{ 1.f, 0.f, 0.f, 1.f };
        vec4 r = rot_y * v0;
        REQUIRE(r.x == Approx(0.f).margin(1e-6f));
        REQUIRE(r.y == Approx(0.f).margin(1e-6f));
        REQUIRE(r.z == Approx(-1.f).margin(epsilon));
        REQUIRE(r.w == Approx(1.f));
    }

    SECTION("Rotation around Z") {
        mat4f mat = mat4f::identity();
        mat4f rot_z = mat.rotate_z(lmath::PI_HALF);
        vec4 v0{ 1.f, 0.f, 0.f, 1.f };
        vec4 r = rot_z * v0;
        REQUIRE(r.x == Approx(0.f).margin(1e-6f));
        REQUIRE(r.y == Approx(1.f).margin(epsilon));
        REQUIRE(r.z == Approx(0.f).margin(1e-6f));
        REQUIRE(r.w == Approx(1.f));
    }
} // TEST_CASE("Matrix operations", "[mat4]")
