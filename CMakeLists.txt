CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(KazemiFaceAlign)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

SET(SOURCES include/train_shape.hpp src/train_shape.cpp src/gb_regression_tree.cpp src/train_face_align.cpp src/facePosePrediction.cpp examples/sample_Landmark_Prediction.cpp)
SET(SOURCES_TRAIN include/train_shape.hpp src/train_shape.cpp src/gb_regression_tree.cpp src/train_face_align.cpp src/facePosePrediction.cpp examples/sample_Face_Train.cpp)
SET(SOURCES_ACCURACY include/train_shape.hpp src/train_shape.cpp src/gb_regression_tree.cpp src/train_face_align.cpp src/facePosePrediction.cpp examples/sample_Test_Accuracy.cpp)
SET(SOURCES_FILTER include/train_shape.hpp src/train_shape.cpp src/gb_regression_tree.cpp src/train_face_align.cpp src/facePosePrediction.cpp examples/sample_Filter.cpp)

ADD_EXECUTABLE(Landmark_prediction_ex ${SOURCES})
ADD_EXECUTABLE(Train_ex ${SOURCES_TRAIN})
ADD_EXECUTABLE(Test_accuracy_ex ${SOURCES_ACCURACY})
ADD_EXECUTABLE(Filter_ex ${SOURCES_FILTER})


FIND_PACKAGE( OpenCV 3 REQUIRED PATHS "/usr/local/include/opencv")

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(Landmark_prediction_ex ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Train_ex ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Test_accuracy_ex ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Filter_ex ${OpenCV_LIBS})