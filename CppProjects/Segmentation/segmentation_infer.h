#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

// #include "algo_data_define.h"

namespace atlas_segmt_dlv3
{
    class ISegmentation
    {
    public:
        ISegmentation(const std::string &model_path, const uint32_t &device_id=0, const uint32_t &model_width=512, const uint32_t &model_height=512);
        ~ISegmentation();

        cv::Mat Excute(const cv::Mat &img);
        cv::Mat Excute(KLImageData &img_data);

    private:
        class Impl;
        std::shared_ptr<Impl> impl_ = nullptr;
    };
}   // atlas_segmt_dlv3