#pragma once

#include <cuda_runtime.h>
#include <LCRUtils/cuda/errcheck.h>

template <typename T>
struct BoundTexture1D {
    cudaArray_t arr;
    cudaTextureObject_t tex;
    size_t size;

    BoundTexture1D(const T* source,
                   const size_t size,
                   const cudaTextureAddressMode addressMode,
                   const cudaTextureFilterMode filterMode,
                   const cudaTextureReadMode readMode = cudaReadModeElementType)
        : size(size) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        CUDA_SAFE_CALL(cudaMallocArray(&arr, &channelDesc, size));
        CUDA_SAFE_CALL(cudaMemcpyToArray(arr, 0, 0, source, size * sizeof(T), cudaMemcpyDeviceToDevice));

        // create texture object
        cudaResourceDesc resourceDescriptor = {};
        resourceDescriptor.resType = cudaResourceTypeArray;
        resourceDescriptor.res.array.array = arr;

        // Setup a texture descriptor
        cudaTextureDesc textureDescriptor = {};
        textureDescriptor.addressMode[0] = addressMode;
        textureDescriptor.filterMode = filterMode;
        textureDescriptor.readMode = readMode;
        textureDescriptor.normalizedCoords = 0;

        // Create the texture object
        CUDA_SAFE_CALL(cudaCreateTextureObject(&tex, &resourceDescriptor, &textureDescriptor, NULL));
    }

    ~BoundTexture1D() {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(tex));
        CUDA_SAFE_CALL(cudaFreeArray(arr));
    }

    BoundTexture1D& operator=(BoundTexture1D const&) = delete;
    BoundTexture1D(const BoundTexture1D& that) = delete;
};

template <typename T>
struct BoundTexture2D {
    cudaArray_t arr;
    cudaTextureObject_t tex;
    int2 size;

    BoundTexture2D(const T* source,
                   const int2 size,
                   const cudaTextureAddressMode addressMode,
                   const cudaTextureFilterMode filterMode,
                   const cudaTextureReadMode readMode = cudaReadModeElementType)
        : size(size) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        CUDA_SAFE_CALL(cudaMallocArray(&arr, &channelDesc, size.x, size.y));
        CUDA_SAFE_CALL(cudaMemcpy2DToArray(arr, 0, 0, source, size.x * sizeof(T), size.x * sizeof(T), size.y, cudaMemcpyDeviceToDevice));

        // create texture object
        cudaResourceDesc resourceDescriptor = {};
        resourceDescriptor.resType = cudaResourceTypeArray;
        resourceDescriptor.res.array.array = arr;

        // Setup a texture descriptor
        cudaTextureDesc textureDescriptor = {};
        textureDescriptor.addressMode[0] = addressMode;
        textureDescriptor.addressMode[1] = addressMode;
        textureDescriptor.filterMode = filterMode;
        textureDescriptor.readMode = readMode;
        textureDescriptor.normalizedCoords = 0;

        // Create the texture object
        CUDA_SAFE_CALL(cudaCreateTextureObject(&tex, &resourceDescriptor, &textureDescriptor, NULL));
    }

    ~BoundTexture2D() {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(tex));
        CUDA_SAFE_CALL(cudaFreeArray(arr));
    }

    BoundTexture2D& operator=(BoundTexture2D const&) = delete;
    BoundTexture2D(const BoundTexture2D& that) = delete;
};

template <typename T, int flag=cudaArrayDefault>
struct BoundTexture3D {
    cudaArray_t arr;
    cudaTextureObject_t tex;
    int3 size;

    BoundTexture3D(const int3 size,
                   const cudaTextureAddressMode addressMode,
                   const cudaTextureFilterMode filterMode,
                   const cudaTextureReadMode readMode = cudaReadModeElementType)
        : size(size) {
        const cudaExtent extent = make_cudaExtent(size.x, size.y, size.z);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
		CUDA_SAFE_CALL(cudaMalloc3DArray(&arr, &channelDesc, extent, flag));

        // create texture object
        cudaResourceDesc resourceDescriptor = {};
        resourceDescriptor.resType = cudaResourceTypeArray;
        resourceDescriptor.res.array.array = arr;

        // Setup a texture descriptor
        cudaTextureDesc textureDescriptor = {};
        textureDescriptor.addressMode[0] = addressMode;
        textureDescriptor.addressMode[1] = addressMode;
        textureDescriptor.addressMode[2] = addressMode;
        textureDescriptor.filterMode = filterMode;
        textureDescriptor.readMode = readMode;
        textureDescriptor.normalizedCoords = 0;

        // Create the texture object
        CUDA_SAFE_CALL(cudaCreateTextureObject(&tex, &resourceDescriptor, &textureDescriptor, NULL));
    }

    void setData(const T* source) {
        const cudaExtent extent = make_cudaExtent(size.x, size.y, size.z);

        cudaMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, extent.width * sizeof(T),
                                                extent.width, extent.height);
        copyParams.dstArray = arr;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToDevice;

        CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
    }

    ~BoundTexture3D() {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(tex));
        CUDA_SAFE_CALL(cudaFreeArray(arr));
    }

    BoundTexture3D& operator=(BoundTexture3D const&) = delete;
    BoundTexture3D(const BoundTexture3D& that) = delete;
};
