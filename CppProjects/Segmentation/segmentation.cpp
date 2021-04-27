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
#include "segmentation.h"
#include <iostream>
#include "model_process.h"
#include "acl/acl.h"
#include "utils.h"
#include "algo_data_define.h"

// #define MODEL_INPUT_WIDTH 500
// #define MODEL_INPUT_HEIGHT 375
#define RGB_IMAGE_SIZE_F32(width, height) ((width) * (height)*3 * 4)
#define IMAGE_CHAN_SIZE_F32(width, height) ((width) * (height)*4)

#define RGB_IMAGE_SIZE_U8(width, height) ((width) * (height)*3)
#define IMAGE_CHAN_SIZE_U8(width, height) ((width) * (height))

using namespace std;

namespace
{
    std::vector<std::tuple<std::string, cv::Scalar>> VocLabel = {
        std::make_tuple("background", cv::Scalar(0, 0, 0)),
        std::make_tuple("aeroplane", cv::Scalar(0, 0, 128)),
        std::make_tuple("bicycle", cv::Scalar(0, 128, 0)),
        std::make_tuple("bird", cv::Scalar(0, 128, 128)),
        std::make_tuple("boat", cv::Scalar(128, 0, 0)),
        std::make_tuple("bottle", cv::Scalar(128, 0, 128)),
        std::make_tuple("bus", cv::Scalar(128, 128, 0)),
        std::make_tuple("car", cv::Scalar(128, 128, 128)),
        std::make_tuple("cat", cv::Scalar(0, 0, 64)),
        std::make_tuple("chair", cv::Scalar(0, 0, 192)),
        std::make_tuple("cow", cv::Scalar(0, 128, 64)),
        std::make_tuple("dining_table", cv::Scalar(0, 128, 192)),
        std::make_tuple("dog", cv::Scalar(128, 0, 64)),
        std::make_tuple("horse", cv::Scalar(128, 0, 192)),
        std::make_tuple("motorbike", cv::Scalar(128, 128, 64)),
        std::make_tuple("person", cv::Scalar(128, 128, 192)),
        std::make_tuple("potted_plant", cv::Scalar(0, 64, 0)),
        std::make_tuple("sheep", cv::Scalar(0, 64, 128)),
        std::make_tuple("sofa", cv::Scalar(0, 192, 0)),
        std::make_tuple("train", cv::Scalar(0, 192, 128)),
        std::make_tuple("monitor", cv::Scalar(128, 64, 0))};
}

namespace atlas_segmt_dlv3
{

    Segmentation::Segmentation(const char *modelPath, const size_t &deviceId, const size_t &modelWidth, const size_t &modelHeight)
        : deviceId_(deviceId), model_width_(modelWidth), model_height_(modelHeight), context_(nullptr), stream_(nullptr), is_inited_(false)
    {
        model_path_ = modelPath;
    }

    Segmentation::~Segmentation()
    {
        DestroyResource();
    }

    Result Segmentation::Init()
    {
        if (is_inited_)
        {
            INFO_LOG("Segmentation instance is initied already!");
            return SUCCESS;
        }

        Result ret = InitResource();
        if (ret != SUCCESS)
        {
            ERROR_LOG("Init acl resource failed");
            return FAILED;
        }

        ret = InitModel(model_path_);
        if (ret != SUCCESS)
        {
            ERROR_LOG("Init model failed");
            return FAILED;
        }

        is_inited_ = true;
        return SUCCESS;
    }

    Result Segmentation::InitResource()
    {
        g_acl_init_cnt ++;
        aclError ret = aclInit(nullptr);
        if (ret != ACL_ERROR_NONE && ret != ACL_ERROR_REPEAT_INITIALIZE)
        {
            ERROR_LOG("acl init failed");
            return FAILED;
        }
        INFO_LOG("acl init success, ret: %d", ret);

        // open device
        ret = aclrtSetDevice(deviceId_);
        if (ret != ACL_ERROR_NONE)
        {
            ERROR_LOG("acl open device %d failed", deviceId_);
            return FAILED;
        }
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

        return SUCCESS;
    }

    Result Segmentation::InitModel(const char *modelPath)
    {
        aclError ret = model_.LoadModelFromFile(modelPath);
        if (ret != SUCCESS)
        {
            ERROR_LOG("execute LoadModelFromFile failed");
            return FAILED;
        }
        ret = model_.CreateDesc();
        if (ret != SUCCESS)
        {
            ERROR_LOG("execute CreateDesc failed");
            return FAILED;
        }

        // create output dataset(for inference result saving) according to modelDesc
        ret = model_.CreateOutput();
        if (ret != SUCCESS)
        {
            ERROR_LOG("execute CreateOutput failed");
            return FAILED;
        }
        return SUCCESS;
    }

    Result Segmentation::Preprocess(std::shared_ptr<ImageDesc> &resizedImage, const cv::Mat &image)
    {
        if (image.empty())
        {
            ERROR_LOG("image is empty!");
            return FAILED;
        }
        //resize image to model size
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(model_width_, model_height_));

        input_width_ = image.cols;
        input_height_ = image.rows;

        //Transform  NHWC
        uint32_t size = RGB_IMAGE_SIZE_U8(model_width_, model_height_);
        Utils::ImageNhwc(resizedImage, resized_image, size);

        aclError ret = aclrtGetRunMode(&runMode_);
        if (ret != ACL_ERROR_NONE)
        {
            ERROR_LOG("acl get run mode failed");
        }

        void *imageDev;
        //copy image data to device
        if (runMode_ == ACL_HOST)
        {
            imageDev = Utils::CopyDataHostToDevice(resizedImage->data.get(), size);
            if (imageDev == nullptr)
            {
                ERROR_LOG("Copy image info to device failed");
                return FAILED;
            }
        }
        else
        {
            imageDev = Utils::CopyDataDeviceToDevice(resizedImage->data.get(), size);
            if (imageDev == nullptr)
            {
                ERROR_LOG("Copy image info to device failed");
                return FAILED;
            }
        }

        resizedImage->size = size;
        resizedImage->data.reset((uint8_t *)imageDev,
                                 [](uint8_t *p) { aclrtFree(p); });
        return SUCCESS;
    }

    Result Segmentation::Inference(const cv::Mat &image, cv::Mat &image_mask)
    {

        // create image_desc for saving input image.
        shared_ptr<ImageDesc> image_data = nullptr;
        MAKE_SHARED_NO_THROW(image_data, ImageDesc);
        if (image_data == nullptr)
        {
            ERROR_LOG("Failed to MAKE_SHARED_NO_THROW for ImageDesc.");
            return FAILED;
        }
        Result ret = Preprocess(image_data, image);
        if (ret != SUCCESS)
        {
            ERROR_LOG("Image Preprocess Error!");
            return FAILED;
        }
        // create input for the model inference.
        ret = model_.CreateInput((void *)image_data->data.get(), image_data->size);
        if (ret != SUCCESS)
        {
            ERROR_LOG("execute CreateInput failed");
            return FAILED;
        }

        // execute inference.
        ret = model_.Execute();
        if (ret != SUCCESS)
        {
            ERROR_LOG("execute inference failed");
            return FAILED;
        }

        aclmdlDataset *modelOutput = model_.GetModelOutputData();
        if (modelOutput == nullptr)
        {
            ERROR_LOG("get model output data failed");
            return FAILED;
        }

        ret = PostProcess(image_mask, modelOutput, model_.GetModelDesc());
        if (ret != SUCCESS)
        {
            ERROR_LOG("PostProcess data failed!");
            return FAILED;
        }

        return SUCCESS;
    }

    Result Segmentation::PostProcess(cv::Mat &mask, aclmdlDataset *modelOutput, aclmdlDesc *modelDesc)
    {
        size_t outDatasetNum = aclmdlGetDatasetNumBuffers(modelOutput);
        INFO_LOG("outDatasetNum: %zu", outDatasetNum);
        if (outDatasetNum != 1)
        {
            ERROR_LOG("outDatasetNum=%zu must be 1", outDatasetNum);
            return FAILED;
        }

        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(modelOutput, 0);
        if (dataBuffer == nullptr)
        {
            ERROR_LOG("get model output aclmdlGetDatasetBuffer failed");
            return FAILED;
        }
        void *dataBufferDev = aclGetDataBufferAddr(dataBuffer);
        if (dataBufferDev == nullptr)
        {
            ERROR_LOG("aclGetDataBufferAddr from dataBuffer failed.");
            return FAILED;
        }

        aclDataType dataType = aclmdlGetOutputDataType(modelDesc, 0);

        INFO_LOG("output dataType: %d", dataType);

        size_t bufferSize = aclGetDataBufferSize(dataBuffer);
        if (bufferSize == 0)
        {
            ERROR_LOG("The 0th dataset buffer size of "
                      "model inference output is 0");
            return FAILED;
        }
        INFO_LOG("Get output buffer size: %zu", bufferSize);
        void *data = nullptr;
        if (runMode_ == ACL_HOST)
        {
            data = Utils::CopyDataDeviceToLocal(dataBufferDev, bufferSize);
            if (data == nullptr)
            {
                ERROR_LOG("Copy inference output to host failed");
                return FAILED;
            }
        }
        else
        {
            data = dataBufferDev;
        }

        cv::Mat output_mask(model_height_, model_width_, CV_32FC1, data);
        if (output_mask.empty())
        {
            ERROR_LOG("output mask is empty!");
            return FAILED;
        }

        // mask = output_mask.clone();
        cv::resize(output_mask, mask, cv::Size(input_width_, input_height_));
        delete[]((uint8_t *)data);

        return SUCCESS;
    }

    void Segmentation::DestroyResource()
    {
        model_.DestroyResource();
        // clear resources.
        aclError ret;
        if (stream_ != nullptr)
        {
            aclrtContext ctx = nullptr;
            if (aclrtGetCurrentContext(&ctx) != 0 || ctx != context_)
                aclrtSetCurrentContext(context_);

            ret = aclrtSynchronizeStream(stream_);
            if (ret != ACL_ERROR_NONE)
            {
                ERROR_LOG("Synchronize Stream failed! Error Code: %d", ret);
            }
            
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

        ret = aclrtResetDevice(deviceId_);
        if (ret != ACL_ERROR_NONE)
        {
            ERROR_LOG("reset device failed");
        }
        INFO_LOG("end to reset device is %d", deviceId_);

        //TODO:
        g_acl_init_cnt --;
        if (g_acl_init_cnt == 0)
        {
            ret = aclFinalize();
            if (ret != ACL_ERROR_NONE)
            {
                ERROR_LOG("finalize acl failed");
            }
            INFO_LOG("end to finalize acl");     
        }
    }

} // namespace atlas_segmt_dlv3