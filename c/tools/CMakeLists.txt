# RegesARC Engine Tools — qpack utilities, conversion scripts, model inspection
# Built as part of the engine build system

cmake_minimum_required(VERSION 3.20)

# ── QPACK tools (quantization packager/unpacker) ────────────────────────
add_executable(qpack
    qpack.c
)
target_include_directories(qpack PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(qpack PRIVATE reges_utils pthread)

# ── QPACK inspect utility (model weight inspection) ────────────────────
add_executable(qpack_inspect
    qpack_inspect.c
)
target_include_directories(qpack_inspect PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(qpack_inspect PRIVATE reges_utils pthread)

# ── Model conversion helper (GGUF ↔ native format) ─────────────────────
add_executable(reges_convert
    convert.c
)
target_include_directories(reges_convert PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(reges_convert PRIVATE reges_utils pthread)

# ── Install tools to bin directory ─────────────────────────────────────
install(TARGETS qpack qpack_inspect reges_convert RUNTIME DESTINATION bin/tools)
