cmake_minimum_required(VERSION 3.21)

project(Glowstone LANGUAGES CXX)

# ---- Standard ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
    # Static CRT
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # Make it reach sub-projects
    set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/dist" CACHE PATH "" FORCE)
endif()

# ---- Output layout ----------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>")

# ---- Vulkan -----------------------------------------------------------------
find_package(Vulkan REQUIRED)

# ---- Fetched dependencies -----------------------------------------------------------
include(FetchContent)

set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(glfw)

# ---- Dear ImGui -------------------------------------------------------------
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/imgui")

add_library(imgui STATIC
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp")

target_include_directories(imgui PUBLIC "${IMGUI_DIR}" "${IMGUI_DIR}/backends")
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_NO_PROTOTYPES)

# ---- Sources ----------------------------------------------------------------
file(GLOB_RECURSE GLOWSTONE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

list(FILTER GLOWSTONE_SOURCES EXCLUDE REGEX "/third_party/")

add_executable(Glowstone ${GLOWSTONE_SOURCES})

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" FILES ${GLOWSTONE_SOURCES})

# ---- Includes ---------------------------------------------------------------
target_include_directories(Glowstone PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party")

# ---- Links ------------------------------------------------------------------
target_link_libraries(Glowstone PRIVATE Vulkan::Vulkan glfw imgui)

if(WIN32)
    target_compile_definitions(Glowstone PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS)

    # Executable icon
    enable_language(RC)
    target_sources(Glowstone PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/resources/Glowstone.rc")
endif()

if(MSVC)
    target_compile_options(Glowstone PRIVATE /Zc:__cplusplus)
endif()

# ---- Shaders ----------------------------------------------------------------
find_program(SLANGC_EXECUTABLE slangc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")

set(ASSET_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(SHADER_SRC_DIR  "${ASSET_DIR}/shaders")
set(SHADER_OUT_DIR  "${CMAKE_BINARY_DIR}/shaders")
set(SHADER_MANIFEST "${SHADER_SRC_DIR}/shaders.json")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SHADER_MANIFEST}")
file(READ "${SHADER_MANIFEST}" _manifest)

string(JSON _defaults ERROR_VARIABLE _err GET "${_manifest}" defaults)
if(_err)
    set(_default_target "")
    set(_default_entry "")
else()
    string(JSON _default_target ERROR_VARIABLE _e GET "${_defaults}" target)
    string(JSON _default_entry  ERROR_VARIABLE _e GET "${_defaults}" entry)
endif()

string(JSON _shader_count ERROR_VARIABLE _err LENGTH "${_manifest}" shaders)
if(_err OR _shader_count EQUAL 0)
    message(FATAL_ERROR "${SHADER_MANIFEST}: 'shaders' must be a non-empty array")
endif()

# Reads a shader member
function(_shader_member out_var object key fallback)
    string(JSON _value ERROR_VARIABLE _e GET "${object}" ${key})
    if(_e)
        set(_value "${fallback}")
    endif()
    set(${out_var} "${_value}" PARENT_SCOPE)
endfunction()

if(SLANGC_EXECUTABLE)
    add_custom_target(Shaders
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUT_DIR}"
        COMMENT "Compiling shaders for install")
endif()

set(_declared_sources "")
math(EXPR _last "${_shader_count} - 1")
foreach(i RANGE ${_last})
    string(JSON _shader GET "${_manifest}" shaders ${i})
    string(JSON _name   GET "${_shader}" name)
    string(JSON _stage  GET "${_shader}" stage)
    _shader_member(_entry  "${_shader}" entry  "${_default_entry}")
    _shader_member(_target "${_shader}" target "${_default_target}")

    list(APPEND _declared_sources "${_name}.slang")
    if(TARGET Shaders)
        add_custom_command(TARGET Shaders POST_BUILD
            COMMAND "${SLANGC_EXECUTABLE}" -target ${_target} -stage ${_stage} -entry ${_entry}
                    -o "${SHADER_OUT_DIR}/${_name}.spv" "${SHADER_SRC_DIR}/${_name}.slang"
            COMMENT "slangc ${_name}.slang -> ${_name}.spv"
            VERBATIM)
    endif()
endforeach()

# ---- Dev --------------------------------------------------------------------
set_target_properties(Glowstone PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# ---- Install ----------------------------------------------
if(SLANGC_EXECUTABLE)
    install(CODE "
        execute_process(
            COMMAND \"${CMAKE_COMMAND}\" --build \"${CMAKE_BINARY_DIR}\"
                    --config \"\${CMAKE_INSTALL_CONFIG_NAME}\" --target Shaders
            RESULT_VARIABLE _shader_result)
        if(NOT _shader_result EQUAL 0)
            message(FATAL_ERROR \"Shader compilation failed; install aborted\")
        endif()")
else()
    install(CODE "message(FATAL_ERROR \"Cannot install without slangc: an install must ship \
prebuilt .spv. Set VULKAN_SDK or SLANGC_EXECUTABLE and reconfigure.\")")
endif()

install(TARGETS Glowstone RUNTIME DESTINATION ".")
install(DIRECTORY "${ASSET_DIR}/models"   DESTINATION "assets")
install(DIRECTORY "${ASSET_DIR}/textures" DESTINATION "assets")
install(DIRECTORY "${SHADER_OUT_DIR}/" DESTINATION "assets/shaders" FILES_MATCHING PATTERN "*.spv")
install(DIRECTORY "${SHADER_SRC_DIR}/" DESTINATION "assets/shaders" FILES_MATCHING PATTERN "*.slang")
install(FILES "${SHADER_MANIFEST}" DESTINATION "assets/shaders")
