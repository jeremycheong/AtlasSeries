#pragma once
#include "utils.h"
#include "acl/acl.h"
#include "common/model_process.h"
#include <memory>
#include <functional>

class LPRecoginition
{
public:
    LPRecoginition(const std::string &refineModelPath, const std::string &lprModelPath,
                   AippMode refineAippMode=BGR_PACKAGE, AippMode lprAippMode=BGR_PACKAGE, uint32_t deviceId = 3,
                   uint32_t refineModelWidth = 120, uint32_t refineModelHeight = 48,
                   uint32_t lprModelWidth = 94, uint32_t lprModelHeight = 24);
    ~LPRecoginition();

    Result Init();
    

    Result Inference(const cv::Mat &srcMat, std::vector<int> &results, std::vector<float> &result_confs);
    Result Inference(const cv::Mat &srcMat, std::string &results, std::vector<float> &confs);

    inline std::vector<std::string> GetLabelText()
    {
        return label_vec_;
    }

private:
    Result InitResource();
    Result InitModel(ModelProcess &model, const char *omModelPath);
    Result CreateModelInputdDataset(ModelProcess &model, void **imageDataBuf, uint32_t &imageDataBufSize);

    Result InferModel(ModelProcess &model, std::vector<std::shared_ptr<float> > &outputs_ptr, std::vector<uint32_t> &output_sizes);
    void *GetInferenceOutputItem(uint32_t &itemDataSize,
                                 aclmdlDataset *inferenceOutput,
                                 uint32_t idx);
    
    Result Preprocess(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &inputDataPtr, void* ImageDataBuf, uint32_t ImageDataBufSize, AippMode aipp_mode);
    
    Result RefinePreprocess(const cv::Mat &refine_srcMat);
    Result LprPreprocess(const cv::Mat &lpr_srcMat);

    Result PreprocessNone(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size, std::function<std::vector<cv::Mat> (const cv::Mat &in_img)> const &preprocess_func);
    Result PreprocessBGR(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size);
    Result PreprocessYUV(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size);

    Result RefinePipline(cv::Mat &alignedMat);
    Result LprPipline(std::vector<int> &clsIdxes, std::vector<float> &clsIdx_confs);

    // Result Decode
    void DestroyResource();

private:
    AippMode refine_aipp_mode_;
    AippMode lpr_aipp_mode_;

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;

    std::string refineModelPath_;
    uint32_t refineModelWidth_;
    uint32_t refineModelHeight_;
    void *refineImageDataBuf_;
    uint32_t refineImageDataBufSize_;
    std::shared_ptr<uint8_t> refine_input_nchw_buf_;
    ModelProcess refine_model_;
    
    std::string lprModelPath_;
    uint32_t lprModelWidth_;
    uint32_t lprModelHeight_;
    void *lprImageDataBuf_;
    uint32_t lprImageDataBufSize_;
    std::shared_ptr<uint8_t> lpr_input_nchw_buf_;
    ModelProcess lpr_model_;

    int imageWidth_;
    int imageHeight_;
    float scale_h_, scale_w_;
    cv::Mat inputImageMat_;

    int refine_tmpl_width_;
    int refine_tmpl_height_;
    std::vector<cv::Point> refine_tmpl_points_;
    int crop_tmpl_width_;
    int crop_tmpl_height_;
    std::vector<cv::Point2f> crop_tmpl_points_;
    std::vector<std::string> label_vec_;

    aclrtRunMode runMode_;
    bool isInited_;
    bool isDeviceSet_;

};