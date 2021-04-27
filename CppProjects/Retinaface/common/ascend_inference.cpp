/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File sample_process.cpp
* Description: handle acl resource
*/
#include "ascend_inference.h"
#include <iostream>
#include <float.h>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "acl/acl.h"
#include "model_process.h"
#include "utils.h"


AscendInference::AscendInference(const std::string &model_path, const uint32_t &model_width,
                    const uint32_t &model_height, const AippMode &aippMode, 
                    const uint32_t &deviceId, aclrtStream stream )
    : deviceId_(deviceId), context_(nullptr), stream_(stream), isInited_(false), isDeviceSet_(false)
{
    ModelPath_ = model_path;
    ImageDataBuf_ = nullptr;
    ModelWidth_ = model_width;
    ModelHeight_ = model_height;

    aipp_mode_ = aippMode;

    switch (aipp_mode_)
    {
    case NONE_PACKAGE:  // 预处理在外部实现
        /* code */
        ImageDataBufSize_ = RGBFP32_IMAGE_SIZE(ModelWidth_, ModelHeight_);
        break;
    
    case BGR_PACKAGE:  // 预处理在模型里面实现，输入可为opencv读入的cv::Mat数据
        ImageDataBufSize_ = BGRU8_IMAGE_SIZE(ModelWidth_, ModelHeight_);
        break;

    case YUV420SP_PACKAGE:  // 预处理在模型里面做，输入为yuv数据
        ImageDataBufSize_ = YUV420SP_IMAGE_SIZE(ModelWidth_, ModelHeight_);
        break;
    
    default:
        WARN_LOG("aippMode input ERROR, use default mode: NONE_PACKAGE");
        ImageDataBufSize_ = RGBFP32_IMAGE_SIZE(ModelWidth_, ModelHeight_);
        break;
    }

    input_nchw_buf_.reset(new uint8_t[ImageDataBufSize_], [](uint8_t* p){delete[](p);});
}

AscendInference::~AscendInference()
{
    DestroyResource();
}

Result AscendInference::InitResource()
{
    // ACL init
    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE && ret != ACL_ERROR_REPEAT_INITIALIZE)
    {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    if (stream_ == nullptr)
    {
        // open device
        ret = aclrtSetDevice(deviceId_);
        if (ret != ACL_ERROR_NONE)
        {
            ERROR_LOG("acl open device %d failed", deviceId_);
            return FAILED;
        }
        isDeviceSet_ = true;
        INFO_LOG("open device %d success", deviceId_);

        // create context (set current)
        ret = aclrtCreateContext(&context_, deviceId_);
        if (ret != ACL_ERROR_NONE)
        {
            ERROR_LOG("acl create context failed");
            return FAILED;
        }
        INFO_LOG("create context success");

        // create stream
        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_ERROR_NONE)
        {
            ERROR_LOG("acl create stream failed");
            return FAILED;
        }
        INFO_LOG("create stream success");
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_ERROR_NONE)
    {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }

    return SUCCESS;
}

Result AscendInference::InitModel(ModelProcess &model, const char *omModelPath)
{
    Result ret = model.LoadModelFromFileWithMem(omModelPath);
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = model.CreateDesc();
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = model.CreateOutput();
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    return SUCCESS;
}

Result AscendInference::CreateModelInputdDataset(ModelProcess &model, void **imageDataBuf, uint32_t &imageDataBufSize)
{
    aclError aclRet = aclrtMalloc(imageDataBuf, imageDataBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_ERROR_NONE)
    {
        ERROR_LOG("malloc device data buffer failed, aclRet is %d", aclRet);
        return FAILED;
    }

    std::vector<void *> input_buffers = {*imageDataBuf};
    std::vector<size_t> input_buffers_size = {imageDataBufSize};
    Result ret = model.CreateInput(input_buffers, input_buffers_size);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Create mode input dataset failed");
        return FAILED;
    }

    return SUCCESS;
}

Result AscendInference::Init()
{
    if (isInited_)
    {
        INFO_LOG("Object detection instance is initied already!");
        return SUCCESS;
    }

    Result ret = InitResource();
    if (ret != SUCCESS)
    {
        ERROR_LOG("Init acl resource failed");
        return FAILED;
    }

    ret = InitModel(model_, ModelPath_.c_str());
    if (ret != SUCCESS)
    {
        ERROR_LOG("Init refine model failed");
        return FAILED;
    }

    // ret = dvpp_.InitResource(stream_);
    // if (ret != SUCCESS)
    // {
    //     ERROR_LOG("Init dvpp failed");
    //     return FAILED;
    // }

    ret = CreateModelInputdDataset(model_, &ImageDataBuf_, ImageDataBufSize_);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Create refine image data buf failed");
        return FAILED;
    }

    isInited_ = true;
    return SUCCESS;
}

Result AscendInference::InferModel(ModelProcess &model, std::vector<std::shared_ptr<float> > &outputs_ptr, std::vector<uint32_t> &output_sizes)
{
    auto ret = model.Execute();
    if (ret != SUCCESS)
    {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    auto inferenceOutput = model.GetModelOutputData();
    size_t outDatasetNum = aclmdlGetDatasetNumBuffers(inferenceOutput);
    // INFO_LOG("outDatasetNum: [%zu]", outDatasetNum);
    for (size_t i = 0; i < outDatasetNum; i ++)
    {
        uint32_t dataSize = 0;
        std::shared_ptr<float> outputData = nullptr;
        outputData.reset((float *)GetInferenceOutputItem(dataSize, inferenceOutput, i), [](float* p){delete[](p);});
        if (!outputData)
        {
            ERROR_LOG("refine model get data ptr is null");
            return FAILED;
        }

        outputs_ptr.push_back(outputData);
        output_sizes.push_back(dataSize / sizeof(float));
    }

    return SUCCESS;
}

Result AscendInference::Preprocess(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &inputDataPtr, void* ImageDataBuf, uint32_t ImageDataBufSize)
{
    if (resizedImg.empty())
    {
        ERROR_LOG("The input image is empty");
        return FAILED;
    }

    switch (aipp_mode_)
    {
    case NONE_PACKAGE:
        /* code */
        PreprocessNone(resizedImg, inputDataPtr, ImageDataBufSize);
        break;
    
    case BGR_PACKAGE:
        PreprocessBGR(resizedImg, inputDataPtr, ImageDataBufSize);
        break;
    
    case YUV420SP_PACKAGE:
        PreprocessYUV(resizedImg, inputDataPtr, ImageDataBufSize);
        break;

    default:
        WARN_LOG("aippMode input ERROR, use default mode: NONE_PACKAGE");
        PreprocessNone(resizedImg, inputDataPtr, ImageDataBufSize);
        break;
    }

    aclrtMemcpyKind policy = (runMode_ == ACL_HOST) ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_DEVICE;

    aclError ret = aclrtMemcpy(ImageDataBuf, ImageDataBufSize,
                               inputDataPtr.get(), ImageDataBufSize, policy);
    if (ret != ACL_ERROR_NONE)
    {
        ERROR_LOG("Copy resized image data to device failed. ERROR code: %d", (int)ret);
        return FAILED;
    }

    return SUCCESS;

}

void *AscendInference::GetInferenceOutputItem(uint32_t &itemDataSize,
                                                    aclmdlDataset *inferenceOutput,
                                                    uint32_t idx)
{
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(inferenceOutput, idx);
    if (dataBuffer == nullptr)
    {
        ERROR_LOG("Get the %dth dataset buffer from model "
                  "inference output failed",
                  idx);
        return nullptr;
    }

    void *dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr)
    {
        ERROR_LOG("Get the %dth dataset buffer address "
                  "from model inference output failed",
                  idx);
        return nullptr;
    }

    size_t bufferSize = aclGetDataBufferSize(dataBuffer);
    if (bufferSize == 0)
    {
        ERROR_LOG("The %dth dataset buffer size of "
                  "model inference output is 0",
                  idx);
        return nullptr;
    }

    void *data = nullptr;
    if (runMode_ == ACL_HOST)
    {
        data = Utils::CopyDataDeviceToLocal(dataBufferDev, bufferSize);
        if (data == nullptr)
        {
            ERROR_LOG("Copy inference output to host failed");
            return nullptr;
        }
    }
    else
    {
        data = dataBufferDev;
    }

    itemDataSize = bufferSize;
    return data;
}


Result AscendInference::PreprocessNone(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size)
{
    cv::Mat resizedMat;
    cv::cvtColor(resizedImg, resizedMat, CV_BGR2RGB);

    cv::Mat resizedMatF32;
    resizedMat.convertTo(resizedMatF32, CV_32FC3, 1.0f / 255);
    std::vector<cv::Mat> rgb_channels;
    cv::split(resizedMatF32, rgb_channels);

    int channel_size = rgb_channels[0].rows * rgb_channels[0].cols * sizeof(float);
    for (size_t i = 0; i < rgb_channels.size(); i ++)
    {
        memcpy(input_data_ptr.get() + (i * channel_size), rgb_channels[i].ptr<float>(0), channel_size);
    }

    return SUCCESS;
}

Result AscendInference::PreprocessBGR(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size)
{
    memcpy(input_data_ptr.get(), resizedImg.data, data_size);

    return SUCCESS;
}

Result AscendInference::PreprocessYUV(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size)
{
    ERROR_LOG("PreprocessYUV is unspport");
    return FAILED;
}

void AscendInference::DestroyResource()
{
    INFO_LOG("DestroyResource");
    model_.DestroyResource();
    // dvpp_.DestroyResource();
    aclError ret;
    if (stream_ != nullptr)
    {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE)
        {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr)
    {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE)
        {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    if (isDeviceSet_)
    {
        ret = aclrtResetDevice(deviceId_);
        if (ret != ACL_ERROR_NONE)
        {
            ERROR_LOG("reset device failed");
        }
        INFO_LOG("end to reset device is %d", deviceId_);
    }

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE)
    {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
}
