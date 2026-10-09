# RegesARC Engine — Top-level CMake build system
# Builds: reges_engine (binary), reges_serve (HTTP lib), per-family shared libs

cmake_minimum_required(VERSION 3.20)
project(reges_engine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Backend selection (CUDA primary, HIP secondary, CPU fallback) ──────────────
option(USE_CUDA "Build with CUDA backend" OFF)
option(USE_HIP   "Build with AMD ROCm/HIP backend" OFF)
option(BUILD_SHARED_FAMILIES "Compile family adapters as shared libraries (.so/.dll)" ON)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(REGES_BACKEND_CUDA=1)
    message(STATUS "CUDA backend enabled")
elseif(USE_HIP)
    find_package(hip REQUIRED)
    add_compile_definitions(REGES_BACKEND_HIP=1)
    message(STATUS "HIP/ROCm backend enabled")
else()
    add_compile_definitions(REGES_BACKEND_CPU=1)
    message(STATUS "CPU fallback backend (no GPU acceleration)")
endif()

# ── Common compile flags ───────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# ── Include directories ────────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# ── Shared utility library (threading, logging, error handling) ────────────────
add_library(reges_utils STATIC
    src/engine.cpp
    src/gguf_loader.cpp
)
target_link_libraries(reges_utils PUBLIC pthread)
if(WIN32)
    target_link_libraries(reges_utils PUBLIC ws2_32)
endif()

# ── HTTP server library (reges_serve) ──────────────────────────────────────────
add_library(reges_serve SHARED
    src/serve_http.cpp
)
target_include_directories(reges_serve PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(WIN32)
    target_link_libraries(reges_serve PRIVATE ws2_32)
else()
    target_link_libraries(reges_serve PUBLIC pthread)
endif()

# ── Family adapter shared libraries ────────────────────────────────────────────
set(REGES_FAMILIES qwen38 deepseek_v4 glm regescore)

foreach(family ${REGES_FAMILIES})
    if(BUILD_SHARED_FAMILIES)
        add_library(${family}_adapter SHARED
            src/family_${family}.cpp
        )
        target_include_directories(${family}_adapter PUBLIC ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(${family}_adapter PRIVATE reges_utils)

        # Export the family registration symbol so the engine can dlopen them
        if(MSVC)
            set_target_properties(${family}_adapter PROPERTIES
                OUTPUT_NAME "reges_${family}"
            )
        else()
            set_target_properties(${family}_adapter PROPERTIES
                PREFIX ""
                SUFFIX ".so"
                OUTPUT_NAME "reges_${family}"
            )
        endif()

        message(STATUS "Building family adapter: ${family} → reges_${family}.so/.dll")
    else()
        # Static build: compile into the main binary
        add_library(${family}_adapter STATIC
            src/family_${family}.cpp
        )
        target_include_directories(${family}_adapter PUBLIC ${CMAKE_SOURCE_DIR}/include)
        target_link_libraries(${family}_adapter PRIVATE reges_utils)
    endif()
endforeach()

# ── Main engine binary (reges_engine) ─────────────────────────────────────────
add_executable(reges_engine
    src/engine.cpp
    src/gguf_loader.cpp
)
target_include_directories(reges_engine PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(reges_engine PRIVATE reges_utils pthread)

if(USE_CUDA)
    target_link_libraries(reges_engine PRIVATE CUDA::cudart CUDA::cuda_driver)
elseif(USE_HIP)
    target_link_libraries(reges_engine PRIVATE hip::host)
endif()

# ── Install targets ────────────────────────────────────────────────────────────
install(TARGETS reges_engine RUNTIME DESTINATION bin)
if(BUILD_SHARED_FAMILIES)
    foreach(family ${REGES_FAMILIES})
        install(TARGETS ${family}_adapter LIBRARY DESTINATION lib/reges_adapters)
    endforeach()
endif()
install(FILES include/reges_api.h DESTINATION include)

# ── Tools subdirectory ─────────────────────────────────────────────────────────
add_subdirectory(tools)
