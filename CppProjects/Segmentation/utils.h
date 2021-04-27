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

* File utils.h
* Description: handle file operations
*/
#pragma once
#include <iostream>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "acl/acl.h"
// #include "opencv2/imgcodecs/legacy/constants_c.h"
#include "algo_data_define.h"
#include "opencv2/imgproc/types_c.h"

namespace atlas_segmt_dlv3 {
#define ERROR_LOG(fmt, args...)                                       \
    {                                                                 \
        char err_buffer[200];                                         \
        sprintf(err_buffer, "\033[31m" fmt "\033[0m", ##args);        \
        KLAlgoLogger::Inst().Log("error", "[{}:{}] {}: {}", __FILE__, \
                                 __LINE__, __FUNCTION__, err_buffer); \
    }

#define INFO_LOG(fmt, args...)                               \
    {                                                        \
        char info_buffer[200];                               \
        sprintf(info_buffer, fmt, ##args);                   \
        KLAlgoLogger::Inst().Log("info", "{}", info_buffer); \
    }

#define WARN_LOG(fmt, args...) \
    {}

using namespace std;

typedef enum Result { SUCCESS = 0, FAILED = 1 } Result;

enum KLImageDevice {
    KL_HOST,
    KL_DEVICE,
};
typedef enum AippMode {
    NONE_PACKAGE = 0,
    BGR_PACKAGE = 1,
    YUV420SP_PACKAGE = 2,
} AippMode;

enum KLImageFormat {
    KL_CV_MAT_BGR888,
    KL_YUV420SP_NV12,
};

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

/**
 * Utils
 */
class Utils {
public:

    static void *CopyDataToDevice(void *data, uint32_t dataSize,
                                  aclrtMemcpyKind policy);

    static void *CopyDataHostToDevice(void *deviceData, uint32_t dataSize);

    static void *CopyDataDeviceToDevice(void *deviceData, uint32_t dataSize);

    static void *CopyDataDeviceToLocal(void *deviceData, uint32_t dataSize);

    static void ImageNhwc(shared_ptr<ImageDesc> &imageData,
                          cv::Mat &nhwcImageChs, uint32_t size);
};

}  // namespace atlas_segmt_dlv3