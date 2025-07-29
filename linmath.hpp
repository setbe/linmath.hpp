#ifndef __LINEAR_MATH_H
#define __LINEAR_MATH_H

#include <stddef.h> // size_t
#include <stdint.h> // uint32_t

#ifndef LINMATH_FREESTANDING
#   include <math.h>
#endif

// -------------------- C++17 feature detection -------------------------------
// MSVC
#if defined(_MSC_VER)
#   define COMPILER_MSVC _MSC_VER

#   if defined(_MSVC_LANG)
#       define CPP_VERSION _MSVC_LANG
#   else
#       define CPP_VERSION __cplusplus
#   endif

#   if CPP_VERSION >= 201703L
#       define LINMATH_CXX_17
#   endif 

// CLang
#elif defined(__clang__)
#   if __cplusplus >= 201703L
#       define LINMATH_CXX_17
#   endif

// GNU
#elif defined(__GNUC__)
#   if __cplusplus >= 201703L
#       define LINMATH_CXX_17
#   endif
#endif // C++17 feature detection

// -------------------- Attribute/constexpr defines ---------------------------

#ifdef LINMATH_CXX_17 // Define `LINMATH_CXX_17` macro if C++17 or above used
#   define LINMATH_NO_DISCARD [[nodiscard]]
#   define LINMATH_CONSTEXPR constexpr
#   define LINMATH_CONSTEXPR_VAR constexpr
#else // fallback
#   define LINMATH_NO_DISCARD
#   define LINMATH_CONSTEXPR inline
#   define LINMATH_CONSTEXPR_VAR static const
#endif // // Attribute/constexpr defines

namespace lmath {
    LINMATH_CONSTEXPR_VAR float PI = 3.14159265359f;
    LINMATH_CONSTEXPR_VAR float PI_HALF = 1.57079632679f;
    LINMATH_CONSTEXPR_VAR float PI2 = 6.28318530718f;

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float radians(float degrees) noexcept {
        return degrees * PI / 180.0f;
    } // radians

#ifdef LINMATH_FREESTANDING
    // ------------------------ Functions decl --------------------------------

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float sinf(float x) noexcept;
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float cosf(float x) noexcept;
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float tanf(float x) noexcept;

    // Only one iteration. ~0.175 ulp.
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float sqrtf(float x) noexcept;

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float floorf(float x) noexcept;

    // ------------------------ Functions impl --------------------------------

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float sinf(float x) noexcept {
        bool flip = false;
        const float x2 = x * x;

        // x to [0, 2pi)
        while (x >= PI2) x -= PI2;
        while (x < 0.0f) x += PI2;

        // [-pi/2, pi/2]
        if (x > PI) {
            x -= PI;
            flip = true;
        }
        if (x > PI_HALF)
            x = PI - x;

        // [-pi/2, pi/2]
        float result = x * (1.0f - x2 / 6.0f + (x2 * x2) / 120.0f);
        return flip ? -result : result;
    } // sinf

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float cosf(float x) noexcept {
        return sinf(x + PI_HALF);
    } // cosf

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float tanf(float x) noexcept {
        const float x2 = x * x;
        // [-pi, pi)
        while (x > PI)
            x -= PI2;
        while (x < -PI)
            x += PI2;

        // tanf(x) ~= x + x^3/3 + 2x^5/15 + 17x^7/315
        return x + x * x2 * (1.0f / 3.0f + x2 * (2.0f / 15.0f + x2 * (17.0f / 315.0f)));
    } // tanf

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float sqrtf(float x) noexcept {
        if (x <= 0.0f) return 0.0f;

        float x_half = 0.5f * x;
        union { float f; uint32_t i; } u = { x }; // reinterpret as int
        u.i = 0x5f3759df - (u.i >> 1); // magic number

        float y = *(float*)&u.i; // TODO: UB
        y = y * (1.5f - x_half * y * y); // 1st Newton-Raphson iteration
        return x * y;
    } // sqrtf

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR float floorf(float x) noexcept {
        int i = (int)x;
        if (x < 0.0f && x != static_cast<float>(i)) --i;
        return static_cast<float>(i);
    } // floorf
#endif // no std

    // ----------------------- Vec Base ---------------------------------------

    // Forward declaration
    template <typename T, size_t N> struct vec;

    // ----------------------- Aliases for vectors ----------------------------

    using vec2 = vec<float, 2>;
    using vec3 = vec<float, 3>;
    using vec4 = vec<float, 4>;

    using ivec2 = vec<int, 2>;
    using ivec3 = vec<int, 3>;
    using ivec4 = vec<int, 4>;

    using uvec2 = vec<unsigned, 2>;
    using uvec3 = vec<unsigned, 3>;
    using uvec4 = vec<unsigned, 4>;

    namespace internal {
        // === Generic vector base class (CRTP) ===
        template <typename Derived, typename T, size_t N>
struct vec_base {
            // Vector addition: this + other
            LINMATH_NO_DISCARD LINMATH_CONSTEXPR
Derived operator+ (const Derived& other) const noexcept {
                const Derived& self = *static_cast<const Derived*>(this);
                Derived r;
                for (size_t i = 0; i < N; ++i)
                    r[i] = self[i] + other[i];
                return r;
            }

            // Vector subtraction: this - other
            LINMATH_NO_DISCARD LINMATH_CONSTEXPR
Derived operator- (const Derived& other) const noexcept {
                const Derived& self = *static_cast<const Derived*>(this);
                Derived r;
                for (size_t i = 0; i < N; ++i)
                    r[i] = self[i] - other[i];
                return r;
            }

            // Scalar multiplication: this * scalar
            LINMATH_NO_DISCARD LINMATH_CONSTEXPR
Derived operator* (T s) const noexcept {
                const Derived& self = *static_cast<const Derived*>(this);
                Derived r;
                for (size_t i = 0; i < N; ++i)
                    r[i] = self[i] * s;
                return r;
            }

            // Dot product of two vectors
            LINMATH_NO_DISCARD
static T dot(const Derived& a, const Derived& b) noexcept {
                T sum{};
                for (size_t i = 0; i < N; ++i)
                    sum += a[i] * b[i];
                return sum;
            }

            // Dot product with self
            LINMATH_NO_DISCARD LINMATH_CONSTEXPR
T dot() const noexcept {
                const Derived& self = *static_cast<const Derived*>(this);
                return vec_base<Derived, T, N>::dot(self, self); // Call to static member
            }

            // Vector magnitude (length)
            LINMATH_NO_DISCARD
T length() const noexcept { return sqrtf(this->dot()); }

            // Normalize vector to unit length
            LINMATH_NO_DISCARD
Derived normalized() const noexcept {
                const Derived& self = *static_cast<const Derived*>(this);
                T len = length();
                return (len > T(0)) ? 
                    (self * (T(1) / len))
                  : Derived{};
            }

            // Component-wise minimum of two vectors
            LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static Derived min(const Derived& a, const Derived& b) noexcept {
                Derived r;
                for (size_t i = 0; i < N; ++i)
                    r[i] = (a[i] < b[i]) ? a[i] : b[i];
                return r;
            }

            // Component-wise maximum of two vectors
            LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static Derived max(const Derived& a, const Derived& b) noexcept {
                Derived r;
                for (size_t i = 0; i < N; ++i)
                    r[i] = (a[i] > b[i]) ? a[i] : b[i];
                return r;
            }
        }; // vec_base
    } // namespace internal

    // ------------------ CRTP N-dimensional Vector ---------------------------
template <typename T, size_t N>
struct 
vec : internal::vec_base<vec<T, N>, T, N> {
    private:
        T v[N]{}; // internal array of values

    public:
        LINMATH_CONSTEXPR
vec() noexcept = default;

        // Constructor from C-style array
        LINMATH_CONSTEXPR
vec(const T(&arr)[N]) noexcept {
            for (size_t i = 0; i < N; ++i)
                v[i] = arr[i];
        }

        // Element access (non-const)
        LINMATH_NO_DISCARD LINMATH_CONSTEXPR
T& operator[] (size_t i) noexcept { return v[i]; }

        // Element access (const)
        LINMATH_NO_DISCARD LINMATH_CONSTEXPR
T operator[] (size_t i) const noexcept { return v[i]; }
    }; // vec

    // ------------------ vec2<T> CRTP Specialization -------------------------
template <typename T>
struct
vec<T, 2> : internal::vec_base<vec<T, 2>, T, 2> {
        union {
            struct { T x, y; };
            T v[2]{};
        };

        // Constructors

        LINMATH_CONSTEXPR vec() noexcept = default;
        LINMATH_CONSTEXPR vec(T x_, T y_) noexcept : x{ x_ }, y{ y_ } {}
        LINMATH_CONSTEXPR vec(const T(&arr)[2]) noexcept { x = arr[0]; y = arr[1]; }

        // Operators []

        LINMATH_NO_DISCARD LINMATH_CONSTEXPR T& operator[](size_t i) noexcept { return v[i]; }
        LINMATH_NO_DISCARD LINMATH_CONSTEXPR T operator[](size_t i) const noexcept { return v[i]; }
    }; // vec2

    // ---------------------- vec3<T> CRTP Specialization ---------------------
template <typename T>
struct
vec<T, 3> : internal::vec_base<vec<T, 3>, T, 3> {
        union {
            struct { T x, y, z; };
            struct { T r, g, b; };
            T v[3]{};
        };

        // Constructors

        LINMATH_CONSTEXPR vec() noexcept = default;
        LINMATH_CONSTEXPR vec(T x_, T y_, T z_) noexcept : x{ x_ }, y{ y_ }, z{ z_ } {}
        LINMATH_CONSTEXPR vec(const T(&arr)[3]) noexcept { x = arr[0]; y = arr[1]; z = arr[2]; }

        // Operators []

        LINMATH_NO_DISCARD LINMATH_CONSTEXPR T& operator[](size_t i) noexcept { return v[i]; }
        LINMATH_NO_DISCARD LINMATH_CONSTEXPR T operator[](size_t i) const noexcept { return v[i]; }

        LINMATH_CONSTEXPR 
static vec<T, 3>
cross(const vec<T, 3>& a, const vec<T, 3>& b) noexcept {
            return { a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x };
        } // cross

        LINMATH_CONSTEXPR
vec<T, 3>
cross(const vec<T, 3>& other) const noexcept {
            return vec<T, 3>::cross(*this, other); // Call static member
        } // cross

        LINMATH_CONSTEXPR
static vec<T, 3>
reflect(const vec<T, 3>& v, const vec<T, 3>& n) noexcept {
            T p = T(2) * vec<T, 3>::dot(v, n);
            return v - n * p;
        } // reflect

        LINMATH_CONSTEXPR
vec<T, 3>
reflect(const vec<T, 3>& normal) const noexcept {
            return vec<T, 3>::reflect(*this, normal); // Call static member
        } // reflect
    }; // vec3

    // ------------------ vec4<T> CRTP Specialization -------------------------
template <typename T>
struct 
vec<T, 4> : internal::vec_base<vec<T, 4>, T, 4> {
        union {
            struct { T x, y, z, w; };
            struct { T r, g, b, a; };
            struct { T left, top, width, height; };
            T v[4]{};
        };

        // Constructors

        LINMATH_CONSTEXPR vec() noexcept = default;
        LINMATH_CONSTEXPR vec(T x_, T y_, T z_, T w_) noexcept : x{ x_ }, y{ y_ }, z{ z_ }, w{ w_ } {}
        LINMATH_CONSTEXPR vec(const T(&arr)[4]) noexcept { x = arr[0]; y = arr[1]; z = arr[2]; w = arr[3]; }

        // Operators []

        LINMATH_NO_DISCARD LINMATH_CONSTEXPR T& operator[](size_t i) noexcept { return v[i]; }
        LINMATH_NO_DISCARD LINMATH_CONSTEXPR T operator[](size_t i) const noexcept { return v[i]; }

        // Misc

        LINMATH_CONSTEXPR
static vec<T, 4>
cross(const vec<T, 4>& a, const vec<T, 4>& b) noexcept {
            return { a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x,
                     T(1) };
        } // cross

        LINMATH_CONSTEXPR
vec<T, 4>
cross(const vec<T, 4>& other) const noexcept {
            return vec<T, 4>::cross(*this, other); // Call static member
        } // cross

        LINMATH_CONSTEXPR
static vec<T, 4>
reflect(const vec<T, 4>& v, const vec<T, 4>& n) noexcept {
            T p = T(2) * vec<T, 4>::dot(v, n);
            vec<T, 4> r;
            for (size_t i = 0; i < 4; ++i)
                r[i] = v[i] - p * n[i];
            return r;
        } // reflect

        LINMATH_CONSTEXPR
vec<T, 4> 
reflect(const vec<T, 4>& normal) const noexcept {
            return vec<T, 4>::reflect(*this, normal); // Call static member
        } // reflect
    }; // vec4


// --------------------------- mat4 -------------------------------------------
template <typename T>
struct
mat4 {
    vec<T, 4> cols[4]{};

    // Direct access
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR vec<T, 4>&       operator[](size_t i) noexcept { return cols[i]; }
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR const vec<T, 4>& operator[](size_t i) const noexcept { return cols[i]; }

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR static mat4 dup(const mat4& n) noexcept { return n; }
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR static vec<T, 4> col(const mat4& m, int i) noexcept { return m[i]; }

    // Identity matrix
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4
identity() noexcept {
        mat4 m{};
        for (int i = 0; i < 4; ++i)
            m[i][i] = T(1);
        return m;
    }

    // Transpose
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4 
transpose(const mat4& n) noexcept {
        mat4 m;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = n[j][i];
        return m;
    }
    
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4<T>
outer_product(const vec<T, 3>& a, const vec<T, 3>& b) noexcept {
        mat4<T> M{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                M[i][j] = (i < 3 && j < 3) ? a[i] * b[j] : T(0);
        return M;
    }

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4
zero() noexcept {
        mat4 m{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = T(0);
        return m;
    }

    // Row extraction
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static vec<T, 4>
row(const mat4& m, int i) noexcept {
        vec<T, 4> r;
        for (int k = 0; k < 4; ++k)
            r[k] = m[k][i];
        return r;
    }

    // Add
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4 
add(const mat4& a, const mat4& b) noexcept {
        mat4 m;
        for (int i = 0; i < 4; ++i)
            m[i] = a[i] + b[i];
        return m;
    }

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
mat4
operator+ (const mat4& other) const noexcept {
        return mat4::add(*this, other); // Call static member
    }

    // Sub
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4 
sub(const mat4& a, const mat4& b) noexcept {
        mat4 m;
        for (int i = 0; i < 4; ++i)
            m[i] = a[i] - b[i];
        return m;
    }

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
mat4
operator- (const mat4& other) const noexcept {
        return mat4::sub(*this, other); // Call static member
    }

    // Matrix multiplication
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4 
mul(const mat4& a, const mat4& b) noexcept {
        mat4 result{};
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                for (int k = 0; k < 4; ++k)
                    result[c][r] += a[k][r] * b[c][k];
        return result;
    }

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
mat4
operator* (const mat4& other) const noexcept {
        return mat4::mul(*this, other); // Call static member
    }

    // Uniform scale
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4 
scale(const mat4& a, T k) noexcept {
        mat4 m;
        for (int i = 0; i < 4; ++i)
            m[i] = a[i] * k;
        return m;
    }

    // Anisotropic scale
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4 
scale_aniso(const mat4& a, T x, T y, T z) noexcept {
        mat4 m;
        m[0] = a[0] * x;
        m[1] = a[1] * y;
        m[2] = a[2] * z;
        m[3] = a[3];
        return m;
    }

    // Matrix * vector
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static vec<T, 4>
mul_vec4(const mat4& m, const vec<T, 4>& v) noexcept {
        vec<T, 4> r;
        for (int j = 0; j < 4; ++j) {
            r[j] = T(0);
            for (int i = 0; i < 4; ++i)
                r[j] += m[i][j] * v[i];
        }
        return r;
    }

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
vec<T, 4>
operator* (const vec<T, 4>& v) const noexcept {
        return mat4::mul_vec4(*this, v); // Call static member
    }

    // Translation matrix
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4
translate(T x, T y, T z) noexcept {
        mat4 t = mat4::identity();
        t[3][0] = x;
        t[3][1] = y;
        t[3][2] = z;
        return t;
    }

    // In-place translation
    LINMATH_CONSTEXPR
void
translate_in_place(T x, T y, T z) noexcept {
        vec<T, 4> t = { x, y, z, T(0) };
        for (int i = 0; i < 4; ++i) {
            vec<T, 4> r = row((*this), i);
            (*this)[3][i] += vec<T, 4>::dot(r, t);
        }
    }

    // Outer product from vec3
    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
static mat4 
from_vec3_mul_outer(const vec<T, 3>& a, const vec<T, 3>& b) noexcept {
        mat4 m{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = (i < 3 && j < 3) ? a[i] * b[j] : T(0);
        return m;
    }

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
mat4 
rotate(float x, float y, float z, float angle) const noexcept {
        float s = sinf(angle);
        float c = cosf(angle);
        vec<T, 3> u = { x, y, z };

        if (u.length() <= 1e-4f)
            return *this;

        u = u.normalized();
        mat4 T = mat4::outer_product(u, u);

        mat4 S = mat4::zero();
        S[0][1] = u.z; S[0][2] = -u.y;
        S[1][0] = -u.z; S[1][2] = u.x;
        S[2][0] = u.y; S[2][1] = -u.x;

        S *= s;

        mat4 C = mat4::identity() - T;
        C *= c;

        mat4 R = T + C + S;
        R[3][3] = 1.f;
        return (*this) * R;
    } // rotate

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
mat4 
rotate_x(float angle) const noexcept {
        float s = sinf(angle);
        float c = cosf(angle);
        mat4 R = { vec<T, 4>{1.f, 0.f, 0.f, 0.f},
                   vec<T, 4>{0.f,   c,   s, 0.f},
                   vec<T, 4>{0.f,  -s,   c, 0.f},
                   vec<T, 4>{0.f, 0.f, 0.f, 1.f} };
        return (*this) * R;
    } // rotate_x

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
mat4 
rotate_y(float angle) const noexcept {
        float s = sinf(angle);
        float c = cosf(angle);
        mat4 R = { vec<T, 4>{   c, 0.f,  -s, 0.f},
                   vec<T, 4>{ 0.f, 1.f, 0.f, 0.f},
                   vec<T, 4>{   s, 0.f,   c, 0.f},
                   vec<T, 4>{ 0.f, 0.f, 0.f, 1.f} };
        return (*this) * R;
    }

    LINMATH_NO_DISCARD LINMATH_CONSTEXPR
mat4
rotate_z(float angle) const noexcept {
        float s = sinf(angle);
        float c = cosf(angle);
        mat4 R = { vec<T, 4>{   c,   s, 0.f, 0.f},
                   vec<T, 4>{  -s,   c, 0.f, 0.f},
                   vec<T, 4>{ 0.f, 0.f, 1.f, 0.f},
                   vec<T, 4>{ 0.f, 0.f, 0.f, 1.f} };
        return (*this) * R;
    }
}; // struct mat4

// ----------------------- Aliases for matrices -------------------------------

using mat4f = mat4<float>;

} // namespace lmath
#endif // __LINEAR_MATH_H