#pragma once

#include <memory>
#include "opencv2/opencv.hpp"
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
// #include "atlas_utils.h"

using namespace std;

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "\033[33m[WARNING]  [%s:%d] %s: " fmt "\n\033[0m", __FILE__, __LINE__, __FUNCTION__, ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "\033[31m[ERROR]  [%s:%d] %s: " fmt "\n\033[0m", __FILE__, __LINE__, __FUNCTION__, ##args)


/**
 * @brief calculate RGB 24bits image size
 * @param [in] width:  image width
 * @param [in] height: image height
 * @return bytes size of image
 */
#define RGBU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)

/**
 * @brief calculate YUVSP420 image size
 * @param [in] width:  image width
 * @param [in] height: image height
 * @return bytes size of image
 */
#define YUV420SP_SIZE(width, height) ((width) * (height) * 3 / 2)

/**
 * @brief generate shared pointer of dvpp memory
 * @param [in] buf: memory pointer, malloc by acldvppMalloc
 * @return shared pointer of input buffer
 */
#define SHARED_PRT_DVPP_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { acldvppFree(p); }))

/**
 * @brief generate shared pointer of memory
 * @param [in] buf memory pointer, malloc by new
 * @return shared pointer of input buffer
 */
#define SHARED_PRT_U8_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { delete[](p); }))

/**
 * @brief calculate aligned number
 * @param [in] num: the original number that to aligned
 * @param [in] align: the align factor 
 * @return the number after aligned
 */
#define ALIGN_UP(num, align) (((num) + (align) - 1) & ~((align) - 1))

/**
 * @brief calculate number align with 2
 * @param [in] num: the original number that to aligned
 * @return the number after aligned
 */
#define ALIGN_UP2(num) ALIGN_UP(num, 2)

/**
 * @brief calculate number align with 16
 * @param [in] num: the original number that to aligned
 * @return the number after aligned
 */
#define ALIGN_UP16(num) ALIGN_UP(num, 16)

/**
 * @brief calculate number align with 128
 * @param [in] num: the original number that to aligned
 * @return the number after aligned
 */
#define ALIGN_UP128(num) ALIGN_UP(num, 128)

struct Resolution {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct CropRect
{
    CropRect()=default;
    CropRect(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
    {
        this->x = x;
        this->y = y;
        this->width = width;
        this->height = height;
    }
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};


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

typedef enum KLImageDevice
{
    KL_HOST,
    KL_DEVICE,
} ImageDevice;

typedef enum KLImageFormat
{
    KL_CV_MAT_BGR888,
    KL_YUV420SP_NV12,
} ImageFormat;

struct KLImageData {
	std::uint32_t width = 0;
	std::uint32_t height = 0;
	std::uint32_t align_width = 0;
	std::uint32_t align_height = 0;
	std::uint32_t size = 0;
	KLImageDevice device = KL_HOST;
	KLImageFormat format = KL_YUV420SP_NV12;
	std::shared_ptr<uint8_t> data;
};

struct BBox
{
    cv::Rect rect;
    float score;
    int cls_id;
};

class Utils
{
public:
    static Result PadResize(const cv::Mat &src_image, const int &dest_long_size, cv::Mat &dest_image, float &scale);
    static Result PadResize(const uint32_t &image_width, const uint32_t &image_height, const uint32_t &dest_width, const uint32_t &dest_height, 
                            CropRect &dest_roi, float &scale);
    static Result Focus(const std::vector<cv::Mat> &rgb_channels, std::vector<cv::Mat> &splited_channels);
    static float Sigmoid(const float &val);
    static void QsortDescentInplace(std::vector<BBox> &obj_boxes, int left, int right);
    static void QsortDescentInplace(std::vector<BBox> &obj_boxes);
    static float intersection_area(const BBox& a, const BBox& b);
    static void NmsSortedBboxes(const std::vector<BBox> &obj_boxes, std::vector<int> &picked, float nms_threshold); 

    static Result CvtMatToYuv420sp(const cv::Mat &src, KLImageData& dest);
    static Result CvtYuv420spToMat(KLImageData &src, cv::Mat &dest);
    static Result SaveDvppOutputData(const char *fileName, const void *devPtr, uint32_t dataSize);

    static void *CopyDataToDevice(void *data, uint32_t dataSize, aclrtMemcpyKind policy);
    static void *CopyDataDeviceToLocal(void *deviceData, uint32_t dataSize);
    static void *CopyDataHostToDevice(void *deviceData, uint32_t dataSize);
    static void *CopyDataDeviceToDevice(void *deviceData, uint32_t dataSize);
};