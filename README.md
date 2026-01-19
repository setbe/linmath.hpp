# linmath.hpp

A **freestanding**, header-only linear algebra library for C++ with optional **runtime SIMD dispatch** (SSE2 / AVX / NEON) and a strict focus on **predictable performance**, **minimal dependencies**, and **portability**.

This project started as an engineering fork and re‑design inspired by **`linmath.h`**, with the goal of providing a modern C++ interface, SIMD acceleration, and freestanding compatibility without sacrificing transparency or control.

---

## Key Features

* **Freestanding-friendly** (no dependency on the C++ standard library math facilities)
* **Header-only**
* **Runtime SIMD detection** with clean scalar fallback
* **SSE2 / AVX / NEON** implementations where available
* **No hidden allocations, no exceptions, no RTTI**
* **Explicit math** — predictable instruction counts and data layout
* **Operator overloading** for ergonomic C++ usage

---

## SIMD Support

SIMD is **optional** and selected **at runtime automatically**:

```cpp
lm::simd::max_level() = lm::simd::runtime_level();
```

Supported paths:

* Scalar (baseline)
* SSE2
* AVX / AVX2 (when available)
* NEON (ARM)

You can also **force-disable SIMD** at compile time:

```cpp
#define LMATH_FORCE_NO_SIMD
```

This makes the library suitable for:

* freestanding kernels
* embedded systems
* deterministic builds
* benchmarking and verification

---

## Benchmarks

All benchmarks were executed with **50,000,000 iterations per test** on MSVC, win10, Intel i5 core-9300H.

### SIMD enabled

```
lm::vec3 dot SIMD        :    17.18 ms
glm::vec3 dot SIMD       :    16.60 ms

lm::mat4 mul SIMD        :  1741.45 ms
glm::mat4 mul SIMD       :  1154.17 ms

lm::mat4 * vec4 SIMD     :   480.87 ms
glm::mat4 * vec4 SIMD    :   480.91 ms
```

### Scalar (SIMD disabled)

```
lm::vec3 dot             :    14.07 ms
glm::vec3 dot            :    13.28 ms
linmath vec3 dot         :    13.00 ms

lm::mat4 mul             :  2650.53 ms
glm::mat4 mul            :  3403.65 ms
linmath mat4 mul         :  2274.86 ms

lm::mat4 * vec4          :   534.78 ms
glm::mat4 * vec4         :   792.45 ms
linmath mat4 * vec4      :   550.45 ms
```

These results highlight:

* Comparable scalar performance with `linmath.h`
* SIMD-enabled `lm` matching or exceeding GLM in critical paths
* Significantly faster `mat4 * vec4` compared to GLM in scalar mode

---

## Contributing Benchmarks

**Please submit your benchmark results** via GitHub Issues.

Use the following **mandatory title format**:

```
bench: Your Operating System. Compiler name. CPU
```

Example:

```
bench: msvc. win10. i5 core-9300H, 2.40 GHz, 4 cores, 8 threads
```

Include:

* CPU model
* OS
* Compiler and version
* SIMD enabled / disabled
* Full benchmark output

---

## API Overview

### `lm::vec`

Vector types:

* `vec2`
* `vec3`
* `vec4`

Features:

* Plain-old-data layout
* SIMD-accelerated dot products
* Length, normalization, cross (vec3)
* Overloaded operators:

```cpp
vec3 a, b;
vec3 c = a + b;
vec3 d = a * 3.0f;
float x = dot(a, b);
```

All operators are thin wrappers around explicit math functions — no hidden temporaries.

---

### `lm::mat`

Matrix types:

* `mat3`
* `mat4`

Features:

* Column-major layout
* Fast `mat4 * mat4`
* Fast `mat4 * vec4`
* Explicit constructors for transforms:

```cpp
mat4 T = mat4_translate(x, y, z);
mat4 R = mat4_rotate_x(angle);
mat4 M = T * R;
```

Operator overloads:

```cpp
mat4 C = A * B;
vec4 v2 = M * v;
```

---

### `lm::quat`

Quaternion type:

* `quat`

Features:

* Rotation construction from axis/angle
* Quaternion multiplication
* Conversion to `mat4`
* Normalization and conjugation

```cpp
quat q = quat_from_axis_angle(axis, angle);
mat4 m = mat4_from_quat(q);
```

---

## Freestanding Design

This library intentionally **does not depend on `<cmath>`**.

Instead, it provides its own minimal math layer:

* `sinf`
* `cosf`
* `tanf`
* `sqrtf` (fast, single-iteration Newton–Raphson)
* `floorf`

Example:

```cpp
float r = lm::radians(90.0f);
float s = lm::sinf(r);
```

This allows usage in:

* freestanding environments
* kernels / bootloaders
* bare-metal and embedded systems
* environments without libc

---

## Operator Overloads — Philosophy

Operator overloads are provided **for ergonomics**, not abstraction.

Rules:

* No dynamic allocation
* No polymorphism
* No exceptions
* No hidden control flow

Every operator maps directly to a known math operation with predictable cost.

---

## Acknowledgements

This project would **not exist** without the original **`linmath.h`** library.

Special thanks to its author(s) for:

* clean, minimal C API
* public-domain / permissive licensing
* making this fork legally and technically possible

This library stands on the shoulders of that work, extending it into a modern C++ and SIMD-capable direction while preserving its original spirit.

---

## License

Same permissive spirit as `linmath.h`.
See LICENSE file for details.
