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
#include "retinaface.h"
#include <iostream>
#include <float.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "opencv2/opencv.hpp"
#include "acl/acl.h"
#include "common/model_process.h"
#include "utils.h"


FaceDetection::FaceDetection(const std::string &model_path, const uint32_t &deviceId)
    : AscendInference(model_path, 960, 960, BGR_PACKAGE, deviceId)
{
    stride_steps_ = {8, 16, 32};

    output_shapes_ = {
        // Class
        {4, int(ModelHeight_ / stride_steps_[0]), int(ModelHeight_ / stride_steps_[0])},  // 8  stride
        {4, int(ModelHeight_ / stride_steps_[1]), int(ModelHeight_ / stride_steps_[1])},    // 16 stride
        {4, int(ModelHeight_ / stride_steps_[2]), int(ModelHeight_ / stride_steps_[2])},    // 32 stride
        // Box
        {8, int(ModelHeight_ / stride_steps_[0]), int(ModelHeight_ / stride_steps_[0])},  // 8  stride
        {8, int(ModelHeight_ / stride_steps_[1]), int(ModelHeight_ / stride_steps_[1])},    // 16 stride
        {8, int(ModelHeight_ / stride_steps_[2]), int(ModelHeight_ / stride_steps_[2])},    // 32 stride
        // Point
        {20, int(ModelHeight_ / stride_steps_[0]), int(ModelHeight_ / stride_steps_[0])},  // 8  stride
        {20, int(ModelHeight_ / stride_steps_[1]), int(ModelHeight_ / stride_steps_[1])},    // 16 stride
        {20, int(ModelHeight_ / stride_steps_[2]), int(ModelHeight_ / stride_steps_[2])},    // 32 stride
    };

    layer_op_.lightmode = true;
    layer_op_.num_threads = 8;

    for (const auto &conv_output_shape : output_shapes_)
    {
        int out_w = conv_output_shape[0] / 2;
        int out_h = -1;     // auto calculate
        int out_c = 1;
        ncnn::ParamDict reshape_pd;
        reshape_pd.set(0, out_w);        
        reshape_pd.set(1, out_h);
        reshape_pd.set(2, out_c);
        reshape_pd.set(3, 1); // before reshape need permute chw -> hwc
        std::shared_ptr<ncnn::Layer> reshape_layer = nullptr;
        reshape_layer.reset(ncnn::create_layer("Reshape"));
        reshape_layer->load_param(reshape_pd);
        reshape_layers_.emplace_back(reshape_layer);
    }

    ncnn::ParamDict softmax_pd;
    softmax_pd.set(0, 2);
    softmax_pd.set(1, 2333);
    softmax_layer_.reset(ncnn::create_layer("Softmax"));
    softmax_layer_->load_param(softmax_pd);

    min_sizes_ = {
        {16, 32}, 
        {64, 128}, 
        {256, 512}
    };

    prob_threshold_ = 0.8f;
    nms_threshold_ = 0.4f;
}

Result FaceDetection::Init()
{
    AscendInference::Init();

    for (size_t i = 0; i < stride_steps_.size(); i ++)
    {
        std::vector<Anchor> anchors;
        GenerateAnchors(min_sizes_[i], stride_steps_[i], anchors);
        if (anchors.empty())
        {
            ERROR_LOG("GenerateAnchors stride[%d] Failed", stride_steps_[i]);
            return FAILED;
        }
        stride_anchors_.emplace_back(anchors);
    }

    INFO_LOG("FaceDetection Init success");

    return SUCCESS;
}

Result FaceDetection::Inference(const cv::Mat &src_mat, std::vector<cv::Rect> &rects, std::vector<float> &confs, 
                                std::vector<std::vector<cv::Point2f> > &points)
{
    rects.clear();
    confs.clear();
    points.clear();

    if (src_mat.empty())
    {
        ERROR_LOG("The input image is empty");
        return FAILED;
    }

    auto ret = RetinafacePreprocess(src_mat);
    if (ret != SUCCESS)
    {
        ERROR_LOG("RetinafacePreprocess ERROR");
        return FAILED;
    }

    ret = RetinafacePipeline(rects, confs, points);
    if (ret != SUCCESS)
    {
        ERROR_LOG("RetinafacePipeline Failed");
        return FAILED;
    }

    return SUCCESS;
}

Result FaceDetection::Inference(ImageData image_data, std::vector<cv::Rect> &rects, std::vector<float> &confs, 
                                std::vector<std::vector<cv::Point2f> > &points)
{
    cv::Mat src_mat = cv::Mat(cv::Size(image_data.width, image_data.height), CV_8UC3, image_data.data.get()).clone();

    return Inference(src_mat, rects, confs, points);
}

FaceDetection::~FaceDetection()
{
    INFO_LOG("Release Retinaface object");
}

Result FaceDetection::RetinafacePreprocess(const cv::Mat &src_mat)
{
    imageWidth_ = src_mat.cols;
    imageHeight_ = src_mat.rows;
    scale_ = 1.f;
    cv::Mat input_mat(cv::Size(ModelWidth_, ModelHeight_), CV_8UC3, cv::Scalar(114, 114, 114));
    int long_side = imageWidth_ > imageHeight_ ? imageWidth_ : imageHeight_;
    if (long_side > (int)ModelWidth_)
    {
        cv::Size target_size;
        if (imageWidth_ > imageHeight_)
        {
            scale_ = imageWidth_ * 1.f / ModelWidth_;
            target_size = cv::Size(ModelWidth_, std::round(imageHeight_ / scale_));
        }
        else
        {
            scale_ = imageHeight_ * 1.f / ModelHeight_;
            target_size = cv::Size(std::round(imageWidth_ / scale_), ModelHeight_);
        }

        // long side resize to model input size
        cv::Mat resized_mat;
        cv::resize(src_mat, resized_mat, target_size);

        // patch to model input size
        cv::Mat input_roi = input_mat(cv::Rect(0, 0, target_size.width, target_size.height));
        resized_mat.copyTo(input_roi);
    }
    else
    {
        // patch to model input size
        cv::Mat input_roi = input_mat(cv::Rect(0, 0, imageWidth_, imageHeight_));
        src_mat.copyTo(input_roi);
    }

    // cv::imwrite("patch.jpg", input_mat);

    // preprocess and push into device memory
    auto ret = Preprocess(input_mat, input_nchw_buf_, ImageDataBuf_, ImageDataBufSize_);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Parent class Preprocess Failed");
        return FAILED;
    }

    return SUCCESS;
}

Result FaceDetection::RetinafacePipeline(std::vector<cv::Rect> &rects, std::vector<float> &confs, std::vector<std::vector<cv::Point2f> > &points)
{
    std::vector<std::shared_ptr<float> > output_ptrs; 
    std::vector<uint32_t> output_sizes;
    auto ret = InferModel(model_, output_ptrs, output_sizes);
    if (ret != SUCCESS)
    {
        ERROR_LOG("InferModel ERROR");
        return FAILED;
    }

    if (output_ptrs.size() != output_sizes.size() || output_ptrs.size() != output_shapes_.size())
    {
        ERROR_LOG("The om model output num[%zu] must be %zu", output_ptrs.size(), output_shapes_.size());
        return FAILED;
    }

    ret = Postprocess(output_ptrs, output_sizes, rects, confs, points);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Retinaface Postprocess Failed");
        return FAILED;
    }
    return SUCCESS;
}

Result FaceDetection::Postprocess(const std::vector<std::shared_ptr<float> > &output_ptrs, 
                                  const std::vector<uint32_t> &output_sizes,
                                  std::vector<cv::Rect> &rects, std::vector<float> &confs, std::vector<std::vector<cv::Point2f> > &points)
{
    std::vector<ncnn::Mat> output_blobs;
    for (size_t i = 0; i < output_ptrs.size(); i ++)
    {
        const auto &output_shape = output_shapes_[i];
        auto &reshape_layer = reshape_layers_[i];
        ncnn::Mat bottom_blob(output_shape[2], output_shape[1], output_shape[0], output_ptrs[i].get());
        ncnn::Mat reshape_top_blob;
        reshape_layer->forward(bottom_blob, reshape_top_blob, layer_op_);
        if (reshape_top_blob.empty())
        {
            ERROR_LOG("Reshape layer forward Failed");
            return FAILED;
        }

        if (i < 3)  // cls prob use softmax
        {
            softmax_layer_->forward_inplace(reshape_top_blob, layer_op_);
        }
        output_blobs.emplace_back(reshape_top_blob);

        // INFO_LOG("output shape: [%d, %d, %d]", reshape_top_blob.c, reshape_top_blob.h, reshape_top_blob.w);
    }

    std::vector<FaceObject> faceproposals;
    // stride 8
    {
        ncnn::Mat &cls_prob_stride8 = output_blobs[0];
        ncnn::Mat &bbox_pred_stride8 = output_blobs[3];
        ncnn::Mat &landmark_stride8 = output_blobs[6];
        std::vector<Anchor> &anchors_stride8 = stride_anchors_[0];
        // int feat_stride8 = stride_steps_[0];

        std::vector<FaceObject> faceobjects8;
        GenerateProposals(anchors_stride8, 
                          cls_prob_stride8, bbox_pred_stride8, landmark_stride8, 
                          prob_threshold_, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }
    
    // stride 16
    {
        ncnn::Mat &cls_prob_stride16 = output_blobs[1];
        ncnn::Mat &bbox_pred_stride16 = output_blobs[4];
        ncnn::Mat &landmark_stride16 = output_blobs[7];
        std::vector<Anchor> &anchors_stride16 = stride_anchors_[1];
        // int feat_stride16 = stride_steps_[1];

        std::vector<FaceObject> faceobjects16;
        GenerateProposals(anchors_stride16, 
                          cls_prob_stride16, bbox_pred_stride16, landmark_stride16, 
                          prob_threshold_, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 32
    {
        ncnn::Mat &cls_prob_stride32 = output_blobs[2];
        ncnn::Mat &bbox_pred_stride32 = output_blobs[5];
        ncnn::Mat &landmark_stride32 = output_blobs[8];
        std::vector<Anchor> &anchors_stride32 = stride_anchors_[2];
        // int feat_stride32 = stride_steps_[2];

        std::vector<FaceObject> faceobjects32;
        GenerateProposals(anchors_stride32, 
                          cls_prob_stride32, bbox_pred_stride32, landmark_stride32, 
                          prob_threshold_, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // sort all proposals by score from highest to lowest
    QsortDescentInplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    NmsSortedBboxes(faceproposals, picked, nms_threshold_);

    int face_count = picked.size();

    cv::Rect image_rect(0, 0, imageWidth_, imageHeight_);
    for (int i = 0; i < face_count; i++)
    {
        const auto &faceobject = faceproposals[picked[i]];
        // const auto &faceobject = faceproposals[i];
        rects.emplace_back(faceobject.rect & image_rect);

        points.push_back(faceobject.landmark);

        confs.push_back(faceobject.prob);

    }

    return SUCCESS;

}

void FaceDetection::GenerateAnchors(const std::vector<int> &min_size, const int &stride_step,
                                    std::vector<Anchor> &anchors)
{
    anchors.clear();
    for (int row = 0; row < int(ModelHeight_ / stride_step); row ++)
    {
        for (int col = 0; col < int(ModelWidth_ / stride_step); col ++)
        {
            for (const auto &msize : min_size)
            {
                Anchor anchor;
                anchor.width = msize * 1.0 / ModelWidth_;
                anchor.height = msize * 1.0 / ModelHeight_;
                anchor.center_x = (col + 0.5) * stride_step / ModelWidth_;
                anchor.center_y = (row + 0.5) * stride_step / ModelHeight_;
                anchors.emplace_back(anchor);
            }
        }
    }
}

void FaceDetection::GenerateProposals(const std::vector<Anchor>& anchors, 
                                      const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, 
                                      float prob_threshold, std::vector<FaceObject>& faceobjects)
{
    // WARN_LOG("score_blob shape: [%d, %d, %d]", score_blob.c, score_blob.h, score_blob.w);
    int output_blob_h = score_blob.h;
    std::vector<float> variances{0.1, 0.2};
    for (int i = 0; i < output_blob_h; i ++)
    {
        FaceObject face_obj;
        float score = score_blob.channel(0).row(i)[1];  // 第二位是人脸置信度
        if (score < prob_threshold)
            continue;
        const auto &prior_anchor_rect = anchors[i];
        const float* bbox_ptr = bbox_blob.channel(0).row(i);
        const float* landmk_ptr = landmark_blob.channel(0).row(i);
        cv::Rect face_rect;
        std::vector<cv::Point2f> landmk;

        float center_x = (prior_anchor_rect.center_x + bbox_ptr[0] * variances[0] * prior_anchor_rect.width) * ModelWidth_ * scale_; // 框的中心点坐标
        float center_y = (prior_anchor_rect.center_y + bbox_ptr[1] * variances[0] * prior_anchor_rect.height) * ModelHeight_ * scale_;
        float bbox_width = (prior_anchor_rect.width * exp(bbox_ptr[2] * variances[1])) * ModelWidth_ * scale_;
        float bbox_height = (prior_anchor_rect.height * exp(bbox_ptr[3] * variances[1])) * ModelHeight_ * scale_;
        face_rect.x = std::floor(center_x - bbox_width / 2);
        face_rect.y = std::floor(center_y - bbox_height / 2);
        face_rect.width = std::floor(bbox_width);
        face_rect.height = std::floor(bbox_height); 
        
        for (int kp = 0; kp < 5; kp ++)
        {
            cv::Point2f point;
            point.x = (prior_anchor_rect.center_x + landmk_ptr[2 * kp + 0] * variances[0] * prior_anchor_rect.width) * ModelWidth_ * scale_;
            point.y = (prior_anchor_rect.center_y + landmk_ptr[2 * kp + 1] * variances[0] * prior_anchor_rect.height) * ModelHeight_ * scale_;
            landmk.emplace_back(point);
        }
        face_obj.prob = score;
        face_obj.rect = std::move(face_rect);
        face_obj.landmark = std::move(landmk);

        faceobjects.emplace_back(face_obj);
    }
    
}

float FaceDetection::IntersectionArea(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void FaceDetection::QsortDescentInplace(std::vector<FaceObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) QsortDescentInplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) QsortDescentInplace(faceobjects, i, right);
        }
    }
}

void FaceDetection::QsortDescentInplace(std::vector<FaceObject>& faceobjects)
{
    if (faceobjects.empty())
        return;

    QsortDescentInplace(faceobjects, 0, faceobjects.size() - 1);
}

void FaceDetection::NmsSortedBboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const FaceObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = IntersectionArea(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
