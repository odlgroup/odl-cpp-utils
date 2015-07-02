set(Headers
    cuda_utils.h
    cutil_math.h
    CuVector.h
    disableThrustWarnings.h
    enableThrustWarnings.h
    geometry.h
    texture.h
    thrustUtils.h)

set(Sources
	dummy.cpp)

include_directories ("${PROJECT_SOURCE_DIR}")
cuda_include_directories ("${PROJECT_SOURCE_DIR}")

CUDA_ADD_LIBRARY(LCRUtilscuda STATIC 
	${Headers}
    ${Sources}) 
