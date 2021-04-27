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
#include "lprecognition.h"
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
    const uint32_t lprOutputWidth = 18;
    const uint32_t lprOutputHeight = 77;

} // namespace

LPRecoginition::LPRecoginition(const std::string &refineModelPath, const std::string &lprModelPath,
                   AippMode refineAippMode, AippMode lprAippMode, uint32_t deviceId,
                   uint32_t refineModelWidth, uint32_t refineModelHeight,
                   uint32_t lprModelWidth, uint32_t lprModelHeight)
    : deviceId_(deviceId), context_(nullptr), stream_(nullptr), isInited_(false), isDeviceSet_(false)
{
    refineModelPath_ = refineModelPath;
    lprModelPath_ = lprModelPath;

    refineImageDataBuf_ = nullptr;
    lprImageDataBuf_ = nullptr;

    refineModelWidth_ = refineModelWidth;
    refineModelHeight_ = refineModelHeight;
    lprModelWidth_ = lprModelWidth;
    lprModelHeight_ = lprModelHeight;
    refine_aipp_mode_ = refineAippMode;
    lpr_aipp_mode_ = lprAippMode;

    switch (refine_aipp_mode_)
    {
    case NONE_PACKAGE:  // 预处理在外部实现
        /* code */
        refineImageDataBufSize_ = RGBFP32_IMAGE_SIZE(refineModelWidth_, refineModelHeight_);
        break;
    
    case BGR_PACKAGE:  // 预处理在模型里面实现，输入可为opencv读入的cv::Mat数据
        refineImageDataBufSize_ = BGRU8_IMAGE_SIZE(refineModelWidth_, refineModelHeight_);
        break;

    case YUV420SP_PACKAGE:  // 预处理在模型里面做，输入为yuv数据
        refineImageDataBufSize_ = YUV420SP_IMAGE_SIZE(refineModelWidth_, refineModelHeight_);
        break;
    
    default:
        WARN_LOG("aippMode input ERROR, use default mode: NONE_PACKAGE");
        refineImageDataBufSize_ = RGBFP32_IMAGE_SIZE(refineModelWidth_, refineModelHeight_);
        break;
    }

    refine_input_nchw_buf_.reset(new uint8_t[refineImageDataBufSize_], [](uint8_t* p){delete[](p);});

    switch (lpr_aipp_mode_)
    {
    case NONE_PACKAGE:  // 预处理在外部实现
        /* code */
        lprImageDataBufSize_ = RGBFP32_IMAGE_SIZE(lprModelWidth_, lprModelHeight_);
        break;
    
    case BGR_PACKAGE:  // 预处理在模型里面实现，输入可为opencv读入的cv::Mat数据
        lprImageDataBufSize_ = BGRU8_IMAGE_SIZE(lprModelWidth_, lprModelHeight_);
        break;

    case YUV420SP_PACKAGE:  // 预处理在模型里面做，输入为yuv数据
        lprImageDataBufSize_ = YUV420SP_IMAGE_SIZE(lprModelWidth_, lprModelHeight_);
        break;
    
    default:
        WARN_LOG("aippMode input ERROR, use default mode: NONE_PACKAGE");
        lprImageDataBufSize_ = RGBFP32_IMAGE_SIZE(lprModelWidth_, lprModelHeight_);
        break;
    }
    lpr_input_nchw_buf_.reset(new uint8_t[lprImageDataBufSize_], [](uint8_t* p){delete[](p);});

    scale_h_ = 1.0f;
    scale_w_ = 1.0f;

    refine_tmpl_width_ = 360;
    refine_tmpl_height_ = 144;
    refine_tmpl_points_ = {
        cv::Point(96, 36),
        cv::Point(288, 36),
        cv::Point(288, 108),
        cv::Point(96, 108)
    };

    crop_tmpl_width_ = 160;
    crop_tmpl_height_ = 40;
    crop_tmpl_points_ = {
        cv::Point2f(0, 0),
        cv::Point2f(crop_tmpl_width_, 0),
        cv::Point2f(crop_tmpl_width_, crop_tmpl_height_),
        cv::Point2f(0, crop_tmpl_height_)
    };

    label_vec_ = {
        "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
        "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
        "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
        "新", "港", "学", "使", "警", "澳", "军", "空", "海", "领",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", "I", "O", "-"
    };
}

LPRecoginition::~LPRecoginition()
{
    DestroyResource();
}

Result LPRecoginition::InitResource()
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

    return SUCCESS;
}

Result LPRecoginition::InitModel(ModelProcess &model, const char *omModelPath)
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

Result LPRecoginition::CreateModelInputdDataset(ModelProcess &model, void **imageDataBuf, uint32_t &imageDataBufSize)
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

Result LPRecoginition::Init()
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

    ret = InitModel(refine_model_, refineModelPath_.c_str());
    if (ret != SUCCESS)
    {
        ERROR_LOG("Init refine model failed");
        return FAILED;
    }

    ret = InitModel(lpr_model_, lprModelPath_.c_str());
    if (ret != SUCCESS)
    {
        ERROR_LOG("Init lpr model failed");
        return FAILED;
    }

    // ret = dvpp_.InitResource(stream_);
    // if (ret != SUCCESS)
    // {
    //     ERROR_LOG("Init dvpp failed");
    //     return FAILED;
    // }

    ret = CreateModelInputdDataset(refine_model_, &refineImageDataBuf_, refineImageDataBufSize_);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Create refine image data buf failed");
        return FAILED;
    }

    ret = CreateModelInputdDataset(lpr_model_, &lprImageDataBuf_, lprImageDataBufSize_);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Create lpr image data buf failed");
        return FAILED;
    }

    isInited_ = true;
    return SUCCESS;
}

Result LPRecoginition::Inference(const cv::Mat &srcMat, std::vector<int> &results, std::vector<float> &result_confs)
{
    inputImageMat_ = srcMat.clone();
    Result ret = SUCCESS;
    cv::Mat alignedMat;

    ret = RefinePreprocess(srcMat);
    ret = RefinePipline(alignedMat);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Run RefinePipline Failed");
        return FAILED;
    }

    // cv::imwrite("align_plate.jpg", alignedMat);
    // alignedMat = srcMat.clone();

    ret = LprPreprocess(alignedMat);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Run LprPreprocess Failed");
        return FAILED;
    }

    
    ret = LprPipline(results, result_confs);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Run LprPipline Failed");
        return FAILED;
    }

    return SUCCESS;
}

Result LPRecoginition::Inference(const cv::Mat &srcMat, std::string &results, std::vector<float> &confs)
{
    std::vector<int> result_idx;
    // std::vector<float> result_confs;
    auto ret = Inference(srcMat, result_idx, confs);
    if (ret != SUCCESS)
    {
        return ret;
    }
    results.clear();
    for (size_t i = 0; i < result_idx.size(); i ++)
    {
        results.append(label_vec_[result_idx[i]]);
    }

    return SUCCESS;
}

Result LPRecoginition::InferModel(ModelProcess &model, std::vector<std::shared_ptr<float> > &outputs_ptr, std::vector<uint32_t> &output_sizes)
{
    auto ret = model.Execute();
    if (ret != SUCCESS)
    {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    auto inferenceOutput = model.GetModelOutputData();
    size_t outDatasetNum = aclmdlGetDatasetNumBuffers(inferenceOutput);
    INFO_LOG("outDatasetNum: [%zu]", outDatasetNum);
    for (size_t i = 0; i < outDatasetNum; i ++)
    {
        uint32_t dataSize = 0;
        std::shared_ptr<float> outputData = nullptr;
        outputData.reset((float *)GetInferenceOutputItem(dataSize, inferenceOutput, 0), [](float* p){delete[](p);});
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

Result LPRecoginition::Preprocess(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &inputDataPtr, void* ImageDataBuf, uint32_t ImageDataBufSize, AippMode aipp_mode)
{
    if (resizedImg.empty())
    {
        ERROR_LOG("The input image is empty");
        return FAILED;
    }

    switch (aipp_mode)
    {
    case NONE_PACKAGE:
        /* code */
        PreprocessNone(resizedImg, inputDataPtr, ImageDataBufSize, [](const cv::Mat &in_img){
            cv::Mat resizedMatF32;
            in_img.convertTo(resizedMatF32, CV_32FC3, 1.0f / 128, -1);
            std::vector<cv::Mat> rgb_channels;
            cv::split(resizedMatF32, rgb_channels);

            return rgb_channels;
        });
        break;
    
    case BGR_PACKAGE:
        PreprocessBGR(resizedImg, inputDataPtr, ImageDataBufSize);
        break;
    
    case YUV420SP_PACKAGE:
        PreprocessYUV(resizedImg, inputDataPtr, ImageDataBufSize);
        break;

    default:
        WARN_LOG("aippMode input ERROR, use default mode: NONE_PACKAGE");
        PreprocessNone(resizedImg, inputDataPtr, ImageDataBufSize, [](const cv::Mat &in_img){
            cv::Mat resizedMatF32;
            in_img.convertTo(resizedMatF32, CV_32FC3, 1.0f / 128, -1);
            std::vector<cv::Mat> rgb_channels;
            cv::split(resizedMatF32, rgb_channels);

            return rgb_channels;
        });
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

void *LPRecoginition::GetInferenceOutputItem(uint32_t &itemDataSize,
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

Result LPRecoginition::RefinePipline(cv::Mat &alignedMat)
{
    cv::Mat g_mat;
    // get g image
    {
        std::vector<std::shared_ptr<float> > outputs_ptr;
        std::vector<uint32_t> output_sizes;
        auto ret = InferModel(refine_model_, outputs_ptr, output_sizes);
        if (ret != SUCCESS)
        {
            ERROR_LOG("RefinePipline InferModel Failed");
            return FAILED;
        }

        if (outputs_ptr.size() != 1)
        {
            ERROR_LOG("Refine model output num = %zu, must be 1", outputs_ptr.size());
            return FAILED;
        }

        float* output_data_ptr = outputs_ptr[0].get();
        std::vector<cv::Point> points;
        for (uint32_t i = 0; i < output_sizes[0] / 2; i ++)
        {
            int point_x = std::floor(output_data_ptr[2 * i] * inputImageMat_.cols);
            int point_y = std::floor(output_data_ptr[2 * i + 1] * inputImageMat_.rows);
            point_x = point_x < 0 ? 0 : point_x;
            point_y = point_y < 0 ? 0 : point_y;
            points.emplace_back(cv::Point(point_x, point_y));
            // INFO_LOG("Refine model out point [%d, %d]", point_x, point_y);
        }


        cv::Mat transform_mat = cv::estimateRigidTransform(points, refine_tmpl_points_, true);

        // INFO_LOG("RefinePipline get Affine transform mat shape: [%d, %d]", transform_mat.rows, transform_mat.cols);
        
        cv::warpAffine(inputImageMat_, g_mat, transform_mat, cv::Size(refine_tmpl_width_, refine_tmpl_height_));
        if (g_mat.empty())
        {
            ERROR_LOG("warpAffine image is empty");
            return FAILED;
        }
    }

    cv::Mat cropped_mat;
    // get cropped image
    {
        auto ret = RefinePreprocess(g_mat);
        if (ret != SUCCESS)
        {
            ERROR_LOG("Run RefinePreprocess Failed");
            return FAILED;
        }

        std::vector<std::shared_ptr<float> > outputs_ptr;
        std::vector<uint32_t> output_sizes;
        ret = InferModel(refine_model_, outputs_ptr, output_sizes);
        if (ret != SUCCESS)
        {
            ERROR_LOG("RefinePipline InferModel Failed");
            return FAILED;
        }

        if (outputs_ptr.size() != 1)
        {
            ERROR_LOG("Refine model output num = %zu, must be 1", outputs_ptr.size());
            return FAILED;
        }

        float* output_data_ptr = outputs_ptr[0].get();
        std::vector<cv::Point2f> points;
        for (uint32_t i = 0; i < output_sizes[0] / 2; i ++)
        {
            float point_x = std::floor(output_data_ptr[2 * i] * g_mat.cols);
            float point_y = std::floor(output_data_ptr[2 * i + 1] * g_mat.rows);
            point_x = point_x < 0 ? 0 : point_x;
            point_y = point_y < 0 ? 0 : point_y;
            points.emplace_back(cv::Point(point_x, point_y));
            // INFO_LOG("Refine model out point [%f, %f]", point_x, point_y);
        }

        // INFO_LOG("RefinePipline get points for crop size: %zu", points.size());

        cv::Mat transform_mat = cv::getPerspectiveTransform(points, crop_tmpl_points_);
        if (transform_mat.empty())
        {
            ERROR_LOG("getPerspectiveTransform mat is empty");
            return FAILED;
        }
        cv::warpPerspective(g_mat, cropped_mat, transform_mat, cv::Size(crop_tmpl_width_, crop_tmpl_height_));
        
        if (cropped_mat.empty())
        {
            ERROR_LOG("get cropped image is empty");
            return FAILED;
        }
        
    }

    alignedMat = std::move(cropped_mat);

    return SUCCESS;
}

Result LPRecoginition::LprPipline(std::vector<int> &clsIdxes, std::vector<float> &clsIdx_confs)
{
    clsIdxes.clear();

    std::vector<std::shared_ptr<float> > outputs_ptr;
    std::vector<uint32_t> output_sizes;
    auto ret = InferModel(lpr_model_, outputs_ptr, output_sizes);
    if (ret != SUCCESS)
    {
        ERROR_LOG("LprPipline InferModel Failed");
        return FAILED;
    }

    if (outputs_ptr.size() != 1)
    {
        ERROR_LOG("LPR model output num = %zu, must be 1", outputs_ptr.size());
        return FAILED;
    }

    float* output_ptr = outputs_ptr[0].get();

    // std::ofstream file_out("./preds_cpp.txt");
    // for (int i = 0; i < (int)lprOutputHeight; i ++)
    // {
    //     const float* ptr = output_ptr + i * (int)lprOutputWidth;
    //     for (int j = 0; j < (int)lprOutputWidth; j ++)
    //     {
    //         file_out << setiosflags(ios::scientific|ios::showpos) << setprecision(18) << ptr[j] << ", ";
    //     }
    //     file_out << std::endl;
    // }


    std::shared_ptr<float> output_data_T = nullptr;
    output_data_T.reset(new float[lprOutputWidth * lprOutputHeight], [](float* p){delete[](p);});
    float* ptr = output_data_T.get();
    for (int i = 0; i < (int)lprOutputWidth; i ++)
    {
        for (int j = 0; j < (int)lprOutputHeight; j ++)
        {
            ptr[i * lprOutputHeight + j] = output_ptr[j * lprOutputWidth + i];
        }
    }
    int output_data_T_h = lprOutputWidth;
    int output_data_T_w = lprOutputHeight;

    std::vector<int> out_preds;
    std::vector<float> out_pred_confs;
    for (int row = 0; row < output_data_T_h; row ++)
    {
        const float* row_out_data_ptr = ptr + row * output_data_T_w;
        std::vector<float> row_data_vec(row_out_data_ptr, row_out_data_ptr + output_data_T_w);
        std::vector<float> row_data_softmax = Utils::Softmax_1D(row_data_vec);

        int max_loc = Utils::ArgMax_1D(row_data_softmax);
        float max_val = row_data_softmax[max_loc];
        out_preds.emplace_back(max_loc);
        out_pred_confs.emplace_back(max_val);
    }

    // std::vector<float> clsIdx_confs;

    int pre_c = out_preds[0];
    if (pre_c != int(label_vec_.size() - 1))
    {
        clsIdxes.push_back(pre_c);
        clsIdx_confs.push_back(out_pred_confs[0]);
    }
    
    for (size_t i = 0; i < out_preds.size(); i ++)
    {
        const auto &c = out_preds[i];
        if (pre_c == c || c == int(label_vec_.size() - 1))
        {
            if (c == int(label_vec_.size() - 1))
                pre_c = c;
            
            continue;
        }

        clsIdxes.push_back(c);
        clsIdx_confs.push_back(out_pred_confs[i]);
        pre_c = c;
    }

    // conf = 1.0f;
    // for (const auto &clsIdx_conf : clsIdx_confs)
    // {
    //     std::cout << clsIdx_conf << ", ";
    //     conf *= clsIdx_conf;
    // }
    // std::cout << std::endl;

    // std::cout << "conf: " << conf << std::endl;

    return SUCCESS;
}

Result LPRecoginition::RefinePreprocess(const cv::Mat &refine_srcMat)
{
    if (refine_srcMat.empty())
    {
        ERROR_LOG("RefinePreprocess The input image is empty");
        return FAILED;
    }

    cv::Mat resizedMat;
    cv::resize(refine_srcMat, resizedMat, cv::Size(refineModelWidth_, refineModelHeight_));

    auto ret = Preprocess(resizedMat, refine_input_nchw_buf_, refineImageDataBuf_, refineImageDataBufSize_, refine_aipp_mode_);
    if (ret != SUCCESS)
    {
        ERROR_LOG("RefinePreprocess Failed");
        return FAILED;
    }

    return SUCCESS;
}

Result LPRecoginition::LprPreprocess(const cv::Mat &lpr_srcMat)
{
    cv::Mat resizedMat;
    cv::resize(lpr_srcMat, resizedMat, cv::Size(lprModelWidth_, lprModelHeight_));

    // cv::imwrite("./aligned_resized.jpg", resizedMat);

    auto ret = Preprocess(resizedMat, lpr_input_nchw_buf_, lprImageDataBuf_, lprImageDataBufSize_, lpr_aipp_mode_);
    if (ret != SUCCESS)
    {
        ERROR_LOG("RefinePreprocess Failed");
        return FAILED;
    }

    return SUCCESS;

}

Result LPRecoginition::PreprocessNone(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size, 
                                      std::function<std::vector<cv::Mat> (const cv::Mat &in_img)> const &preprocess_func)
{
    std::vector<cv::Mat> rgb_channels = preprocess_func(resizedImg);

    int channel_size = rgb_channels[0].rows * rgb_channels[0].cols * sizeof(float);
    for (size_t i = 0; i < rgb_channels.size(); i ++)
    {
        memcpy(input_data_ptr.get() + (i * channel_size), rgb_channels[i].ptr<float>(0), channel_size);
    }

    return SUCCESS;
}

Result LPRecoginition::PreprocessBGR(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size)
{
    memcpy(input_data_ptr.get(), resizedImg.data, data_size);

    return SUCCESS;
}

Result LPRecoginition::PreprocessYUV(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size)
{
    ERROR_LOG("PreprocessYUV is unspport");
    return FAILED;
}

void LPRecoginition::DestroyResource()
{
    INFO_LOG("DestroyResource");
    refine_model_.DestroyResource();
    lpr_model_.DestroyResource();
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
