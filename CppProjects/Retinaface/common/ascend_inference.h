#pragma once
#include "utils.h"
#include "acl/acl.h"
#include "common/model_process.h"
#include <memory>

class AscendInference
{
public:
    AscendInference(const std::string &model_path, const uint32_t &model_width,
                    const uint32_t &model_height, const AippMode &aippMode=BGR_PACKAGE, 
                    const uint32_t &deviceId = 3, aclrtStream stream = nullptr);

    virtual ~AscendInference();

    virtual Result Init();

    inline Result GetDeviceId(uint32_t &device_id)
    {
        device_id = deviceId_;
        return SUCCESS;
    }

    inline Result GetContext(aclrtContext* context)
    {
        *context = context_;
        return SUCCESS;
    }

    inline Result GetStream(aclrtStream* stream)
    {
        *stream = stream_;
        return SUCCESS;
    }
    
protected:
    Result Preprocess(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &inputDataPtr, void* ImageDataBuf, uint32_t ImageDataBufSize);

    Result InferModel(ModelProcess &model, std::vector<std::shared_ptr<float> > &outputs_ptr, std::vector<uint32_t> &output_sizes);

    // Result Decode
    void DestroyResource();

private:
    Result InitResource();
    Result InitModel(ModelProcess &model, const char *omModelPath);
    Result CreateModelInputdDataset(ModelProcess &model, void **imageDataBuf, uint32_t &imageDataBufSize);
    void *GetInferenceOutputItem(uint32_t &itemDataSize,
                                 aclmdlDataset *inferenceOutput,
                                 uint32_t idx);

    Result PreprocessNone(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size);
    Result PreprocessBGR(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size);
    Result PreprocessYUV(const cv::Mat &resizedImg, std::shared_ptr<uint8_t> &input_data_ptr, uint32_t data_size);

protected:
    uint32_t ModelWidth_;
    uint32_t ModelHeight_;
    void *ImageDataBuf_;
    uint32_t ImageDataBufSize_;
    std::shared_ptr<uint8_t> input_nchw_buf_;
    ModelProcess model_;

private:
    AippMode aipp_mode_;
    std::string ModelPath_;

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;

    aclrtRunMode runMode_;
    bool isInited_;
    bool isDeviceSet_;

};