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
#include "object_detect_yolov5_bgr.h"
#include <iostream>
#include <float.h>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "acl/acl.h"
#include "model_process.h"
#include "utils.h"

using namespace std;

namespace
{
    const uint32_t out80DataBufId = 0;
    const uint32_t out40DataBufId = 1;
    const uint32_t out20DataBufId = 2;

} // namespace

ObjectDetectYolov5BGR::ObjectDetectYolov5BGR(const char *modelPath, bool is_use_aipp, uint32_t deviceId,
                                             uint32_t modelWidth,
                                             uint32_t modelHeight)
    : deviceId_(deviceId), context_(nullptr), stream_(nullptr), isInited_(false), isDeviceSet_(false)
{
    modelWidth_ = (modelWidth);
    modelHeight_ = (modelHeight);

    is_use_aipp_ = is_use_aipp;

    if (is_use_aipp_)   // yuv输入
    {
        modelChannels_ = 1;
        // imageDataBufSize_ = (modelWidth_) * (modelHeight_) * modelChannels_;
        int modelInputAlignedWidth = ALIGN_UP16(modelWidth_);
        int modelInputAlignedHeight = ALIGN_UP2(modelHeight_);
        imageDataBufSize_ = YUV420SP_SIZE(modelInputAlignedWidth, modelInputAlignedHeight);
    }
    else
    {
        modelChannels_ = 12;
        imageDataBufSize_ = (modelWidth_ / 2) * (modelHeight_ / 2) * modelChannels_ * sizeof(float);

        // modelChannels_ = 3;
        // imageDataBufSize_ = (modelWidth_) * (modelHeight_) * modelChannels_ * sizeof(float);
    }

    input_nchw_buf_.reset(new uint8_t[imageDataBufSize_], [](uint8_t* p){delete[](p);});
    // output_data_buf_ = nullptr;
    scale_ = 1.0f;
    prob_threshold_ = 0.25f;
    nms_threshold_ = 0.45f;
    anchors_s8_  = {10.f, 13.f, 16.f, 30.f, 33.f, 23.f};
    anchors_s16_ = {30.f, 61.f, 62.f, 45.f, 59.f, 119.f};
    anchors_s32_ = {116.f, 90.f, 156.f, 198.f, 373.f, 326.f};
    modelPath_ = modelPath;

    dvpp_processor_ = nullptr;
}

ObjectDetectYolov5BGR::~ObjectDetectYolov5BGR()
{
    DestroyResource();
}

Result ObjectDetectYolov5BGR::InitResource()
{
    // ACL init
    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE && ret != ACL_ERROR_REPEAT_INITIALIZE)
    {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

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

    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_ERROR_NONE)
    {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }

    dvpp_processor_ = std::make_shared<DvppCropAndPaste>(stream_, modelWidth_, modelHeight_);
    dvpp_processor_->InitResource();

    return SUCCESS;
}

Result ObjectDetectYolov5BGR::InitModel(const char *omModelPath)
{
    Result ret = model_.LoadModelFromFileWithMem(omModelPath);
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = model_.CreateDesc();
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = model_.CreateOutput();
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    return SUCCESS;
}

Result ObjectDetectYolov5BGR::CreateModelInputdDataset()
{
    aclError aclRet = aclrtMalloc(&imageDataBuf_, imageDataBufSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_ERROR_NONE)
    {
        ERROR_LOG("malloc device data buffer failed, aclRet is %d", aclRet);
        return FAILED;
    }

    std::vector<void *> input_buffers = {imageDataBuf_};
    std::vector<size_t> input_buffers_size = {imageDataBufSize_};
    Result ret = model_.CreateInput(input_buffers, input_buffers_size);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Create mode input dataset failed");
        return FAILED;
    }

    return SUCCESS;
}

Result ObjectDetectYolov5BGR::SetAnchors(const std::vector<std::vector<float> > &anchors)
{
    if (anchors.size() != 3)
    {
        ERROR_LOG("Set anchors size must be 3");
        return FAILED;
    }
    anchors_s8_  = anchors[0];
    anchors_s16_ = anchors[1];
    anchors_s32_ = anchors[2];

    return SUCCESS;
}

Result ObjectDetectYolov5BGR::SetThresholds(const float &object_thr, const float &nms_thr)
{
    prob_threshold_ = object_thr;
    nms_threshold_ = nms_thr;
    return SUCCESS;
}

Result ObjectDetectYolov5BGR::Init()
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

    ret = InitModel(modelPath_);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Init model failed");
        return FAILED;
    }

    // ret = dvpp_.InitResource(stream_);
    // if (ret != SUCCESS)
    // {
    //     ERROR_LOG("Init dvpp failed");
    //     return FAILED;
    // }

    ret = CreateModelInputdDataset();
    if (ret != SUCCESS)
    {
        ERROR_LOG("Create image info buf failed");
        return FAILED;
    }

    isInited_ = true;
    return SUCCESS;
}


Result ObjectDetectYolov5BGR::Preprocess(const cv::Mat &srcMat)
{
    if (srcMat.empty())
    {
        ERROR_LOG("The input image is empty");
        return FAILED;
    }
    imageWidth_ = srcMat.cols;
    imageHeight_ = srcMat.rows;

    cv::Mat rgbMat;
    cv::cvtColor(srcMat, rgbMat, CV_BGR2RGB);
    cv::Mat resizedMat;

    Timer pre_process_timer;
    Utils::PadResize(rgbMat, modelWidth_, resizedMat, scale_);
    int64_t cost_time = pre_process_timer.elapsed();
    WARN_LOG("PadResize cose time: %lu ms",cost_time);

    if (resizedMat.empty())
    {
        ERROR_LOG("Resize image failed");
        return FAILED;
    }

    if (is_use_aipp_)
    {
        memcpy(input_nchw_buf_.get(), resizedMat.data, imageDataBufSize_);
    }
    else
    {
        cv::Mat resizedMatF32;
        resizedMat.convertTo(resizedMatF32, CV_32FC3, 1.0f / 255);
        std::vector<cv::Mat> rgb_channels;
        cv::split(resizedMatF32, rgb_channels);

        std::vector<cv::Mat> input_channels;

        input_channels = rgb_channels;
        // pre_process_timer.reset();
        // Utils::Focus(rgb_channels, input_channels);
        // cost_time = pre_process_timer.elapsed();
        // WARN_LOG("Focus cose time: %lu ms",cost_time);

        int channel_size = input_channels[0].rows * input_channels[0].cols * sizeof(float);
        for (size_t i = 0; i < input_channels.size(); i ++)
        {
            memcpy(input_nchw_buf_.get() + (i * channel_size), input_channels[i].ptr<float>(0), channel_size);
        }
    }

    aclrtMemcpyKind policy = (runMode_ == ACL_HOST) ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_DEVICE_TO_DEVICE;

    aclError ret = aclrtMemcpy(imageDataBuf_, imageDataBufSize_,
                               input_nchw_buf_.get(), imageDataBufSize_, policy);
    if (ret != ACL_ERROR_NONE)
    {
        ERROR_LOG("Copy resized image data to device failed. error code: %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result ObjectDetectYolov5BGR::Preprocess(const KLImageData &srcImageData)
{
    if (srcImageData.device != KL_DEVICE || srcImageData.format != KL_YUV420SP_NV12)
    {
        ERROR_LOG("srcImageData must on NPU and format is KL_YUV420SP_NV12");
        return FAILED;
    }

    if (!srcImageData.data.get())
    {
        ERROR_LOG("The input image is empty");
        return FAILED;
    }

    imageWidth_ = srcImageData.width;
    imageHeight_ = srcImageData.height;

    dvpp_processor_->SetInputSize(imageWidth_, imageHeight_);

    CropRect paste_rect;
    Utils::PadResize(imageWidth_, imageHeight_, modelWidth_, modelHeight_, paste_rect, scale_);
    KLImageData imageDataResized;

    dvpp_processor_->SetPasteRoi(paste_rect);
    // INFO_LOG("dvpp_processor CropAndPasteProcess");
    dvpp_processor_->CropAndPasteProcess(srcImageData, imageDataResized);
    // INFO_LOG("dvpp_processor CropAndPasteProcess done");

    assert(imageDataBufSize_ == imageDataResized.size);

    aclError ret = aclrtMemcpy(imageDataBuf_, imageDataBufSize_,
                            imageDataResized.data.get(), imageDataBufSize_, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_ERROR_NONE)
    {
        ERROR_LOG("Copy resized image data to device failed. error code: %d", ret);
        return FAILED;
    }

    INFO_LOG("New Preprocess success");
    return SUCCESS;
}


Result ObjectDetectYolov5BGR::Inference(std::vector<BBox> &results)
{       
    Timer infer_timer;
    auto ret = model_.Execute();
    if (ret != SUCCESS)
    {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }
    int64_t cost_time = infer_timer.elapsed();
    WARN_LOG("model Execute cose time: %lu ms", cost_time);
    infer_timer.reset();

    auto inferenceOutput = model_.GetModelOutputData();

    size_t outDatasetNum = aclmdlGetDatasetNumBuffers(inferenceOutput);
    // INFO_LOG("outDatasetNum: [%zu]", outDatasetNum);
    if (outDatasetNum != 3) {
        ERROR_LOG("outDatasetNum=%zu must be 3",outDatasetNum);
        return FAILED;
    }

    uint32_t dataSize = 0;
    std::vector<BBox> proposals;

    // stride = 8, feature map shape: 3 x 80 x 80 x (cls + 5)
    {
        std::shared_ptr<float> outputData = nullptr;
        outputData.reset((float *)GetInferenceOutputItem(dataSize, inferenceOutput, out80DataBufId), [](float* p){delete[](p);});
        if (!outputData)
        {
            ERROR_LOG("out80DataBufId: %u get data ptr is null", out80DataBufId);
            return FAILED;
        }
        // INFO_LOG("out80DataBufId: %u, dataSize: %u", out80DataBufId, dataSize);
        size_t num_anchors = anchors_s8_.size() / 2;
        size_t stride = 8;
        size_t num_grid_y = modelHeight_ / stride;
        size_t num_grid_x = modelWidth_ / stride;
        size_t num_classes = dataSize / (num_anchors * num_grid_y * num_grid_x * sizeof(float)) - 5;
        
        std::vector<BBox> obj_boxes;
        GenerateProposals(anchors_s8_, stride, num_grid_y, num_grid_x, num_classes, outputData.get(), obj_boxes);
        proposals.insert(proposals.end(), obj_boxes.begin(), obj_boxes.end());
    }

    // stride = 16, feature map shape: 3 x 40 x 40 x (cls + 5)
    {
        std::shared_ptr<float> outputData = nullptr;
        outputData.reset((float *)GetInferenceOutputItem(dataSize, inferenceOutput, out40DataBufId), [](float* p){delete[](p);});
        if (!outputData)
        {
            ERROR_LOG("out40DataBufId: %u get data ptr is null", out40DataBufId);
            return FAILED;
        }
        // INFO_LOG("out40DataBufId: %u, dataSize: %u", out40DataBufId, dataSize);
        size_t num_anchors = anchors_s16_.size() / 2;
        size_t stride = 16;
        size_t num_grid_y = modelHeight_ / stride;
        size_t num_grid_x = modelWidth_ / stride;
        size_t num_classes = dataSize / (num_anchors * num_grid_y * num_grid_x * 4) - 5;
        std::vector<BBox> obj_boxes;
        GenerateProposals(anchors_s16_, stride, num_grid_y, num_grid_x, num_classes, outputData.get(), obj_boxes);
        proposals.insert(proposals.end(), obj_boxes.begin(), obj_boxes.end());
    }

    // stride = 32, feature map shape: 3 x 20 x 20 x (cls + 5)
    {
        std::shared_ptr<float> outputData = nullptr;
        outputData.reset((float *)GetInferenceOutputItem(dataSize, inferenceOutput, out20DataBufId), [](float* p){delete[](p);});
        if (!outputData)
        {
            ERROR_LOG("out20DataBufId: %u get data ptr is null", out20DataBufId);
            return FAILED;
        }
        // INFO_LOG("out20DataBufId: %u, dataSize: %u", out20DataBufId, dataSize);
        size_t num_anchors = anchors_s32_.size() / 2;
        size_t stride = 32;
        size_t num_grid_y = modelHeight_ / stride;
        size_t num_grid_x = modelWidth_ / stride;
        size_t num_classes = dataSize / (num_anchors * num_grid_y * num_grid_x * 4) - 5;
        
        std::vector<BBox> obj_boxes;
        GenerateProposals(anchors_s32_, stride, num_grid_y, num_grid_x, num_classes, outputData.get(), obj_boxes);
        proposals.insert(proposals.end(), obj_boxes.begin(), obj_boxes.end());
    }

    Utils::QsortDescentInplace(proposals);
    std::vector<int> picked;
    Utils::NmsSortedBboxes(proposals, picked, nms_threshold_);

    size_t count = picked.size();
    results.resize(count);
    for (unsigned int i = 0; i < count; i ++)
    {
        results[i] = proposals[picked[i]];

        results[i].rect.x *= scale_;
        results[i].rect.y *= scale_;
        results[i].rect.width *= scale_;
        results[i].rect.height *= scale_;
        results[i].rect &= cv::Rect(0, 0, imageWidth_, imageHeight_);
    }

    cost_time = infer_timer.elapsed();
    WARN_LOG("post process cose time: %lu ms", cost_time);

    return SUCCESS;
}

void *ObjectDetectYolov5BGR::GetInferenceOutputItem(uint32_t &itemDataSize,
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

Result ObjectDetectYolov5BGR::GenerateProposals(const std::vector<float> &anchors, size_t stride, 
                                                size_t num_grid_y, size_t num_grid_x, size_t num_classes, float* out_data_ptr, 
                                                std::vector<BBox> &obj_boxes)
{
    auto num_anchors = anchors.size() / 2;
    int item_size = num_classes + 5;

    for (size_t q = 0; q < num_anchors; q ++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const float* feat_ptr = out_data_ptr + (q * item_size * num_grid_y * num_grid_x);
        int iter_gap = num_grid_y * num_grid_x;

        // from Reshape layer output
        for (size_t i = 0; i < num_grid_y; i ++)
        {
            for (size_t j = 0; j < num_grid_x; j ++)
            {
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (unsigned int k = 0; k < num_classes; k ++)
                {
                    float score = feat_ptr[(5 + k) * iter_gap + i * num_grid_x + j];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }
                float box_score = feat_ptr[4 * iter_gap + i * num_grid_x + j];

                float confidence = num_classes == 1 ? Utils::Sigmoid(box_score) : Utils::Sigmoid(box_score) * Utils::Sigmoid(class_score);
                
                if (confidence < prob_threshold_)
                    continue;
                // yolov5/models/yolo.py Detect forward
                // y = x[i].sigmoid()
                // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                float dx = Utils::Sigmoid(feat_ptr[0 * iter_gap + i * num_grid_x + j]);
                float dy = Utils::Sigmoid(feat_ptr[1 * iter_gap + i * num_grid_x + j]);
                float dw = Utils::Sigmoid(feat_ptr[2 * iter_gap + i * num_grid_x + j]);
                float dh = Utils::Sigmoid(feat_ptr[3 * iter_gap + i * num_grid_x + j]);

                float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                float pb_w = pow(dw * 2.f, 2) * anchor_w;
                float pb_h = pow(dh * 2.f, 2) * anchor_h;

                // yolo 坐标转Rect坐标
                float x0 = pb_cx - pb_w * 0.5f;
                float y0 = pb_cy - pb_h * 0.5f;
                float x1 = pb_cx + pb_w * 0.5f;
                float y1 = pb_cy + pb_h * 0.5f;

                BBox obj_box;
                obj_box.rect.x = x0;
                obj_box.rect.y = y0;
                obj_box.rect.width = x1 - x0;
                obj_box.rect.height = y1 - y0;
                obj_box.cls_id = class_index;
                obj_box.score = confidence;

                obj_boxes.emplace_back(obj_box);
            }
        }

        // from Transpose layer output

        // for (size_t i = 0; i < num_grid_y; i ++)
        // {
        //     for (size_t j = 0; j < num_grid_x; j ++)
        //     {
        //         const float* feat_ptr = out_data_ptr + (q * num_grid_y * num_grid_x * item_size) + (i * num_grid_x * item_size) + (j * item_size);
        //         int class_index = 0;
        //         float class_score = -FLT_MAX;
        //         for (unsigned int k = 0; k < num_classes; k ++)
        //         {
        //             float score = feat_ptr[5 + k];
        //             if (score > class_score)
        //             {
        //                 class_index = k;
        //                 class_score = score;
        //             }
        //         }
        //         float box_score = feat_ptr[4];

        //         float confidence = num_classes == 1 ? Utils::Sigmoid(box_score) : Utils::Sigmoid(box_score) * Utils::Sigmoid(class_score);
                
        //         if (confidence < prob_threshold_)
        //             continue;
        //         // yolov5/models/yolo.py Detect forward
        //         // y = x[i].sigmoid()
        //         // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
        //         // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

        //         float dx = Utils::Sigmoid(feat_ptr[0]);
        //         float dy = Utils::Sigmoid(feat_ptr[1]);
        //         float dw = Utils::Sigmoid(feat_ptr[2]);
        //         float dh = Utils::Sigmoid(feat_ptr[3]);

        //         float pb_cx = (dx * 2.f - 0.5f + j) * stride;
        //         float pb_cy = (dy * 2.f - 0.5f + i) * stride;

        //         float pb_w = pow(dw * 2.f, 2) * anchor_w;
        //         float pb_h = pow(dh * 2.f, 2) * anchor_h;

        //         // yolo 坐标转Rect坐标
        //         float x0 = pb_cx - pb_w * 0.5f;
        //         float y0 = pb_cy - pb_h * 0.5f;
        //         float x1 = pb_cx + pb_w * 0.5f;
        //         float y1 = pb_cy + pb_h * 0.5f;

        //         BBox obj_box;
        //         obj_box.rect.x = x0;
        //         obj_box.rect.y = y0;
        //         obj_box.rect.width = x1 - x0;
        //         obj_box.rect.height = y1 - y0;
        //         obj_box.cls_id = class_index;
        //         obj_box.score = confidence;

        //         obj_boxes.emplace_back(obj_box);
        //     }
        // }
    }

    return SUCCESS;
}

void ObjectDetectYolov5BGR::DestroyResource()
{
    INFO_LOG("DestroyResource");
    model_.DestroyResource();
    dvpp_processor_->DestroyCropAndPasteResource();
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
        ERROR_LOG("finalize acl failed, error code: %d", ret);
    }
    INFO_LOG("end to finalize acl");
}
