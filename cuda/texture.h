#pragma once

#include <cuda_runtime.h>

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
        cudaMallocArray(&arr, &channelDesc, size);
        cudaMemcpy(arr, source, size, cudaMemcpyDeviceToDevice);

        // create texture object
        cudaResourceDesc resourceDescriptor = {0};
        resourceDescriptor.resType = cudaResourceTypeArray;
        resourceDescriptor.res.array.array = arr;

        // Setup a texture descriptor
        cudaTextureDesc textureDescriptor = {0};
        textureDescriptor.addressMode[0] = addressMode;
        textureDescriptor.filterMode = filterMode;
        textureDescriptor.readMode = readMode;
        textureDescriptor.normalizedCoords = 0;

        // Create the texture object
        cudaCreateTextureObject(&tex, &resourceDescriptor, &textureDescriptor, NULL);
    }

    ~BoundTexture1D() {
        cudaDestroyTextureObject(tex);
        cudaFreeArray(arr);
    }
};

template <typename T>
struct BoundTexture3D {
    cudaArray_t arr;
    cudaTextureObject_t tex;
    int3 size;

    BoundTexture3D(const T* source,
                   const int3 size,
                   const cudaTextureAddressMode addressMode,
                   const cudaTextureFilterMode filterMode,
                   const cudaTextureReadMode readMode = cudaReadModeElementType)
        : size(size) {
        const cudaExtent extent = make_cudaExtent(size.x, size.y, size.z);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        cudaMalloc3DArray(&arr, &channelDesc, extent);

        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)source, extent.width * sizeof(T),
                                                extent.width, extent.height);
        copyParams.dstArray = arr;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParams);

        // create texture object
        cudaResourceDesc resourceDescriptor = {0};
        resourceDescriptor.resType = cudaResourceTypeArray;
        resourceDescriptor.res.array.array = arr;

        // Setup a texture descriptor
        cudaTextureDesc textureDescriptor = {0};
        textureDescriptor.addressMode[0] = addressMode;
        textureDescriptor.addressMode[1] = addressMode;
        textureDescriptor.addressMode[2] = addressMode;
        textureDescriptor.filterMode = filterMode;
        textureDescriptor.readMode = readMode;
        textureDescriptor.normalizedCoords = 0;

        // Create the texture object
        cudaCreateTextureObject(&tex, &resourceDescriptor, &textureDescriptor, NULL);
    }

    ~BoundTexture3D() {
        cudaDestroyTextureObject(tex);
        cudaFreeArray(arr);
    }
};