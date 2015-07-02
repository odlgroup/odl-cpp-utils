set(Headers
    SlowTest.h
    TestUtils.h)

set(Sources
	dummy.cpp)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

CUDA_ADD_LIBRARY(LCRUtilstest STATIC 
	${Headers}
    ${Sources}) 

set_property(TARGET LCRUtilstest PROPERTY FOLDER LCRUtils)