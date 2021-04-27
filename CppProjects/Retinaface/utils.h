#pragma once

#include <memory>
#include "opencv2/opencv.hpp"
#include "acl/acl.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "\033[33m[ERROR]  [%s:%d] %s: " fmt "\n\033[0m", __FILE__, __LINE__, __FUNCTION__, ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "\033[31m[ERROR]  [%s:%d] %s: " fmt "\n\033[0m", __FILE__, __LINE__, __FUNCTION__, ##args)

#define BGRU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)
#define RGBFP32_IMAGE_SIZE(width, height) ((width) * (height) * 3 * sizeof(float))
#define YUV420SP_IMAGE_SIZE(width, height) ((width) * (height) * 3 / 2)

#define ALIGN_UP(num, align) (((num) + (align)-1) & ~((align)-1))
#define ALIGN_UP2(num) ALIGN_UP(num, 2)
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP128(num) ALIGN_UP(num, 128)

typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef enum AippMode
{
    NONE_PACKAGE = 0,
    BGR_PACKAGE = 1,
    YUV420SP_PACKAGE = 2,
} AippMode;

typedef enum ImageDevice
{
    HOST,
    DEVICE,
} ImageDevice;

typedef enum ImageFormat
{
    CV_MAT_BGR888,
    YUV420SP_NV12,
} ImageFormat;

struct ImageData
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t alignWidth = 0;
    uint32_t alignHeight = 0;
    uint32_t size = 0;
    ImageDevice device = HOST;
    ImageFormat format = YUV420SP_NV12;
    std::shared_ptr<uint8_t> data;
};

struct BBox
{
    cv::Rect rect;
    float score;
    int cls_id;
};

struct FaceObject
{
    cv::Rect rect;
    std::vector<cv::Point2f> landmark;
    float prob;
};

class Utils
{
public:
    static Result PadResize(const cv::Mat &src_image, const int &dest_long_size, cv::Mat &dest_image, float &scale);
    static Result Focus(const std::vector<cv::Mat> &rgb_channels, std::vector<cv::Mat> &splited_channels);
    static float Sigmoid(const float &val);
    static int ArgMax_1D(const std::vector<float> &data);
    static void QsortDescentInplace(std::vector<BBox> &obj_boxes, int left, int right);
    static void QsortDescentInplace(std::vector<BBox> &obj_boxes);
    static float intersection_area(const BBox& a, const BBox& b);
    static void NmsSortedBboxes(const std::vector<BBox> &obj_boxes, std::vector<int> &picked, float nms_threshold); 

    static void *CopyDataToDevice(void *data, uint32_t dataSize, aclrtMemcpyKind policy);
    static void *CopyDataDeviceToLocal(void *deviceData, uint32_t dataSize);
    static void *CopyDataHostToDevice(void *deviceData, uint32_t dataSize);
    static void *CopyDataDeviceToDevice(void *deviceData, uint32_t dataSize);
};