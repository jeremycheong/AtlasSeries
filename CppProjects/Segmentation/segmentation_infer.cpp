#include "segmentation_infer.h"

#include "segmentation.h"

namespace atlas_segmt_dlv3
{
class ISegmentation::Impl
{
public:
    Impl(const std::string &model_path, const uint32_t &device_id=0, const uint32_t &model_width=512, const uint32_t &model_height=512)
    {
        segmt_ptr_.reset(new Segmentation(model_path.c_str(), device_id, model_width, model_height));
        segmt_ptr_->Init();
    }

    ~Impl()
    {

    }

    cv::Mat Excute(const cv::Mat &img)
    {
        cv::Mat image_mask;

        Result ret = segmt_ptr_->Inference(img, image_mask);
        if (ret != SUCCESS)
        {
            ERROR_LOG("Segmentation Inference failed");
            return cv::Mat();
        }

        return image_mask;
    }

private:
    std::unique_ptr<Segmentation> segmt_ptr_ = nullptr;

};

ISegmentation::ISegmentation(const std::string &model_path, const uint32_t &device_id, const uint32_t &model_width, const uint32_t &model_height)
{
    impl_.reset(new Impl(model_path, device_id, model_width, model_height));
}

ISegmentation::~ISegmentation()
{
}

cv::Mat ISegmentation::Excute(const cv::Mat &img)
{
    return impl_->Excute(img);
}

cv::Mat ISegmentation::Excute(KLImageData &img_data)
{
    cv::Mat img(cv::Size(img_data.width, img_data.height), CV_8UC3, img_data.data.get());
    return impl_->Excute(img);
}

}