cmake_minimum_required(VERSION 3.16)
project(openvino_yolo26_seg CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force new CXX11 ABI so symbol mangling matches the system OpenCV libraries.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

set(OV_TOOLKIT "openvino_libs/openvino_toolkit_ubuntu22_2025.3.0.19807.44526285f24_x86_64")
set(OpenVINO_DIR "${OV_TOOLKIT}/runtime/cmake" CACHE PATH "OpenVINO cmake config directory")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
add_executable(openvino_yolo26_seg
    src/main.cpp
    src/yolo.cpp
)

target_include_directories(openvino_yolo26_seg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OV_TOOLKIT}/runtime/include
)

target_link_libraries(openvino_yolo26_seg PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
)

# Point the runtime loader at the toolkit's shared libraries.
set_target_properties(openvino_yolo26_seg PROPERTIES
    BUILD_RPATH "${OV_TOOLKIT}/runtime/lib/intel64"
    INSTALL_RPATH "${OV_TOOLKIT}/runtime/lib/intel64"
)

target_compile_options(openvino_yolo26_seg PRIVATE -O3 -march=native -Wall)

# ---------------------------------------------------------------------------
# run_test — multi-threaded efficiency benchmark
# ---------------------------------------------------------------------------
add_executable(run_test
    src/run_test.cpp
    src/yolo.cpp
)

target_include_directories(run_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OV_TOOLKIT}/runtime/include
)

target_link_libraries(run_test PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
    Threads::Threads
)

set_target_properties(run_test PROPERTIES
    BUILD_RPATH "${OV_TOOLKIT}/runtime/lib/intel64"
    INSTALL_RPATH "${OV_TOOLKIT}/runtime/lib/intel64"
)

target_compile_options(run_test PRIVATE -O3 -march=native -Wall)
