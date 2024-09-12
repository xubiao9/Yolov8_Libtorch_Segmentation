CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
project(YOLOv5)


# SET (OpenCV_DIR path/to/opencv/build)  #opencv root
# SET (ONNXRUNTIME_DIR path/to/onnxruntime)
SET(ONNXRUNTIME_DIR /home/xb/ALL/python_pip/onnxruntime-linux-x64-1.9.0)
set(Torch_DIR /home/xb/libtorch/share/cmake/Torch)
# set(OpenCV_DIR "/usr/local/include/opencv4")
set(OpenCV_DIR "/home/xb/ALL/python_pip/opencv-4.9.0/build")
# set (CMAKE_CXX_COMPILER "E:/MinGW/libexec/gcc/mingw32/6.3.0/lto-wrapper.exe")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/xb/libtorch/share/cmake/Torch")

FIND_PACKAGE(OpenCV REQUIRED)
#include_directories("")
ADD_EXECUTABLE(YOLOv5 yolov5.h  yolov5_seg.h  yolov5_utils.h
			 main.cpp yolov5.cpp  yolov5_seg.cpp  yolov5_utils.cpp)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# 引入头文件
include_directories(/home/xb/ALL/python_pip/onnxruntime-linux-x64-1.9.0/include
"/home/xb/libtorch/include"
"/usr/local/include/opencv4"
.)
# 引入库文件
link_directories(/home/xb/ALL/python_pip/onnxruntime-linux-x64-1.9.0/lib)

TARGET_INCLUDE_DIRECTORIES(YOLOv5 PRIVATE "${ONNXRUNTIME_DIR}/include")

TARGET_COMPILE_FEATURES(YOLOv5 PRIVATE cxx_std_14)
TARGET_LINK_LIBRARIES(YOLOv5 ${OpenCV_LIBS})

if (WIN32)
    TARGET_LINK_LIBRARIES(YOLOv5 "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)

if (UNIX)
    TARGET_LINK_LIBRARIES(YOLOv5 "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)