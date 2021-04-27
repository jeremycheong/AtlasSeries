#pragma once
#include "utils.h"
#include "acl/acl.h"
#include "model_process.h"
#include "dvpp_cropandpaste.h"
#include <memory>

#include "timer.hpp"

class ObjectDetectYolov5BGR
{
public:
    ObjectDetectYolov5BGR(const char *modelPath, bool is_use_aipp=false, uint32_t deviceId = 0,
                          uint32_t modelWidth = 640, uint32_t modelHeight = 640);
    ~ObjectDetectYolov5BGR();

    Result Init();
    Result Preprocess(const cv::Mat &srcMat);
    /**
     * @brief npu上图片数据预处理
     * 
     * @param srcImageData 数据在npu上且format为 PIXEL_FORMAT_YUV_SEMIPLANAR_420
     * @return Result 
     */
    Result Preprocess(const KLImageData &srcImageData);

    Result SetAnchors(const std::vector<std::vector<float> > &anchors);
    Result SetThresholds(const float &object_thr=0.25f, const float &nms_thr=0.45f);
    Result Inference(std::vector<BBox> &results);

private:
    Result InitResource();
    Result InitModel(const char *omModelPath);
    Result CreateModelInputdDataset();
    void *GetInferenceOutputItem(uint32_t &itemDataSize,
                                 aclmdlDataset *inferenceOutput,
                                 uint32_t idx);
    Result GenerateProposals(const std::vector<float> &anchors, size_t stride, 
                             size_t num_grid_y, size_t num_grid_x, size_t num_classes, float* out_data_ptr, 
                             std::vector<BBox> &obj_boxes);

    void DestroyResource();

private:
    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;

    void *imageDataBuf_;
    uint32_t imageDataBufSize_;
    std::shared_ptr<uint8_t> input_nchw_buf_;
    bool is_use_aipp_;

    ModelProcess model_;
    std::shared_ptr<DvppCropAndPaste> dvpp_processor_;

    const char *modelPath_;
    uint32_t modelWidth_;
    uint32_t modelHeight_;
    uint32_t modelChannels_;
    int imageWidth_;
    int imageHeight_;
    float scale_;

    float prob_threshold_;
    float nms_threshold_;
    std::vector<float> anchors_s8_;
    std::vector<float> anchors_s16_;
    std::vector<float> anchors_s32_;

    aclrtRunMode runMode_;
    bool isInited_;
    bool isDeviceSet_;

};