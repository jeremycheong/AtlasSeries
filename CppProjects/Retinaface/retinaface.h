#pragma once
#include "utils.h"
#include "acl/acl.h"
#include "common/model_process.h"
#include "common/ascend_inference.h"
#include <memory>

#include "ncnn/layer.h"

class FaceDetection : public AscendInference
{
public:
    using AscendInference::AscendInference;

    FaceDetection(const std::string &model_path, const uint32_t &deviceId = 3);

    Result Init() final;

    Result Inference(const cv::Mat &src_mat, std::vector<cv::Rect> &rects, std::vector<float> &confs, std::vector<std::vector<cv::Point2f> > &points);

    Result Inference(ImageData image_data, std::vector<cv::Rect> &rects, std::vector<float> &confs, std::vector<std::vector<cv::Point2f> > &points);

    ~FaceDetection() final;

private:
    struct Anchor
    {
        float center_x;
        float center_y;
        float width;
        float height;
    };

private:
    Result RetinafacePreprocess(const cv::Mat &src_mat);

    Result RetinafacePipeline(std::vector<cv::Rect> &rects, std::vector<float> &confs, std::vector<std::vector<cv::Point2f> > &points);

    Result Postprocess(const std::vector<std::shared_ptr<float> > &output_ptrs, const std::vector<uint32_t> &output_sizes,
                        std::vector<cv::Rect> &rects, std::vector<float> &confs, std::vector<std::vector<cv::Point2f> > &points);

    void GenerateAnchors(const std::vector<int> &min_size, const int &stride_step, 
                         std::vector<Anchor> &anchors);

    void GenerateProposals(const std::vector<Anchor>& anchors, 
                           const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, 
                           float prob_threshold, std::vector<FaceObject>& faceobjects);

    float IntersectionArea(const FaceObject& a, const FaceObject& b);
    void QsortDescentInplace(std::vector<FaceObject>& faceobjects, int left, int right);
    void QsortDescentInplace(std::vector<FaceObject>& faceobjects);
    void NmsSortedBboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold);

private:
    int imageWidth_;
    int imageHeight_;
    float scale_;

    std::vector<std::vector<int> > min_sizes_;
    std::vector<int> stride_steps_;
    std::vector<std::vector<Anchor> > stride_anchors_;

    float prob_threshold_;
    float nms_threshold_;

    std::vector<std::vector<int> > output_shapes_;

    ncnn::Option layer_op_;

    std::vector<std::shared_ptr<ncnn::Layer> > reshape_layers_;

    std::shared_ptr<ncnn::Layer> softmax_layer_;

};