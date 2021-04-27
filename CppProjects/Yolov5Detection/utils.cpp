#include "utils.h"

Result Utils::PadResize(const cv::Mat &src_image, const int &dest_long_size, cv::Mat &dest_image, float &scale)
{
    int image_w = src_image.cols;
    int image_h = src_image.rows;
    scale = 1.f;
    dest_image = cv::Mat(cv::Size(dest_long_size, dest_long_size), CV_8UC3, cv::Scalar(114, 114, 114));
    int long_side = image_w > image_h ? image_w : image_h;
    if (long_side > (int)dest_long_size)
    {
        cv::Size target_size;
        if (image_w > image_h)
        {
            scale = image_w * 1.f / dest_long_size;
            target_size = cv::Size(dest_long_size, std::round(image_h / scale));
        }
        else
        {
            scale = image_h * 1.f / dest_long_size;
            target_size = cv::Size(std::round(image_w / scale), dest_long_size);
        }

        // long side resize to model input size
        cv::Mat resized_mat;
        cv::resize(src_image, resized_mat, target_size);

        // patch to model input size
        cv::Mat input_roi = dest_image(cv::Rect(0, 0, target_size.width, target_size.height));
        resized_mat.copyTo(input_roi);
    }
    else
    {
        // patch to model input size
        cv::Mat input_roi = dest_image(cv::Rect(0, 0, image_w, image_h));
        src_image.copyTo(input_roi);
    }

    return SUCCESS;
}

Result Utils::PadResize(const uint32_t &image_width, const uint32_t &image_height, const uint32_t &dest_width, const uint32_t &dest_height, 
                            CropRect &dest_roi, float &scale)
{
    scale = 1.0f;
    if (image_width <= (int)dest_width && image_height <= (int)dest_height)
    {
        dest_roi.x = 0;
        dest_roi.y = 0;
        dest_roi.width = image_width;
        dest_roi.height = image_height;
    }
    else if (float(image_width) / image_height >= float(dest_width) / dest_height)  // 按照width缩放
    {
        scale = float(image_width) / dest_width;
        dest_roi.x = 0;
        dest_roi.y = 0;
        dest_roi.width = dest_width;
        dest_roi.height = std::round(image_height / scale);
    }
    else    // 按照height缩放
    {
        scale = float(image_height) / dest_height;
        dest_roi.x = 0;
        dest_roi.y = 0;
        dest_roi.width = std::round(image_width / scale);
        dest_roi.height = dest_height;
    }

    return SUCCESS;
}

Result Utils::Focus(const std::vector<cv::Mat> &rgb_channels, std::vector<cv::Mat> &splited_channels)
{
    splited_channels.clear();
    splited_channels.resize(rgb_channels.size() * 4);
    if (rgb_channels[0].type() != CV_32FC1 || rgb_channels[1].type() != CV_32FC1 || rgb_channels[2].type() != CV_32FC1)
    {
        ERROR_LOG("The bgr_channels type must be CV_32FC1");
        return FAILED;
    }
    int dest_width = rgb_channels[0].cols / 2;
    int dest_height = rgb_channels[0].rows / 2;
    for (size_t ch = 0; ch < rgb_channels.size(); ch ++)
    {
        const cv::Mat &ch_mat = rgb_channels[ch];

        cv::Mat dest_mat00(dest_width, dest_height, CV_32FC1);
        cv::Mat dest_mat01(dest_width, dest_height, CV_32FC1);
        cv::Mat dest_mat10(dest_width, dest_height, CV_32FC1);
        cv::Mat dest_mat11(dest_width, dest_height, CV_32FC1);
        for (int row = 0; row < dest_height; row ++)
        {
            for (int col = 0; col < dest_width; col ++)
            {
                dest_mat00.at<float>(row, col) = ch_mat.at<float>(2* row + 0, 2 * col + 0);
                dest_mat01.at<float>(row, col) = ch_mat.at<float>(2* row + 0, 2 * col + 1);
                dest_mat10.at<float>(row, col) = ch_mat.at<float>(2* row + 1, 2 * col + 0);
                dest_mat11.at<float>(row, col) = ch_mat.at<float>(2* row + 1, 2 * col + 1);
            }
        }
        // 这里排列顺序对类别的confidence影响较大
        splited_channels[0 + ch] = std::move(dest_mat00);
        splited_channels[6 + ch] = std::move(dest_mat01);
        splited_channels[3 + ch] = std::move(dest_mat10);
        splited_channels[9 + ch] = std::move(dest_mat11);
    }

    return SUCCESS;
}

float Utils::Sigmoid(const float &val)
{
    return 1.0 / (1.0 + exp(-val));
}

void Utils::QsortDescentInplace(std::vector<BBox> &obj_boxes, int left, int right)
{
    int i = left;
    int j = right;
    float p = obj_boxes[(left + right) / 2].score;

    while (i <= j)
    {
        while (obj_boxes[i].score > p)
            i++;

        while (obj_boxes[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(obj_boxes[i], obj_boxes[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) QsortDescentInplace(obj_boxes, left, j);
        }
        #pragma omp section
        {
            if (i < right) QsortDescentInplace(obj_boxes, i, right);
        }
    }
}

void Utils::QsortDescentInplace(std::vector<BBox> &obj_boxes)
{
    if (obj_boxes.empty())
        return;

    QsortDescentInplace(obj_boxes, 0, obj_boxes.size() - 1);
}

float Utils::intersection_area(const BBox& a, const BBox& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void Utils::NmsSortedBboxes(const std::vector<BBox> &obj_boxes, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = obj_boxes.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = obj_boxes[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const BBox& a = obj_boxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const BBox& b = obj_boxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

Result Utils::CvtMatToYuv420sp(const cv::Mat &src, KLImageData& dest)
{
    aclrtRunMode runMode;
    aclrtGetRunMode(&runMode);

    int cols = ALIGN_UP16(src.cols);
    int rows = ALIGN_UP2(src.rows);
    cv::Mat YUV_data;
    YUV_data.create(rows + rows / 2, cols, CV_8UC1);

    // #pragma omp parallel num_threads(6)
    for (int i = 0; i < rows; i ++)
    {
        for (int j = 0; j < cols; j ++)
        {
            uchar* YPointer = YUV_data.ptr<uchar>(i);
            int B = src.at<cv::Vec3b>(i, j)[0];
            int G = src.at<cv::Vec3b>(i, j)[1];
            int R = src.at<cv::Vec3b>(i, j)[2];
            int Y = (77 * R + 150 * G + 29 * B) >> 8;
            YPointer[j] = Y;
            uchar* UVPointer = YUV_data.ptr<uchar>(rows+i/2);
            if (i % 2 == 0 && (j) % 2 == 0)
            {
                int U = ((-44 * R - 87 * G + 131 * B) >> 8) + 128;
                int V = ((131 * R - 110 * G - 21 * B) >> 8) + 128;
                UVPointer[j] = U;
                UVPointer[j+1] = V;
            }
        }
    }

    dest.width = src.cols;
    dest.height = src.rows;
    dest.align_width = cols;
    dest.align_height = rows;
    dest.size = YUV420SP_SIZE(cols, rows);

    void* data_buffer_dev = nullptr;
    aclError ret = acldvppMalloc(&data_buffer_dev, dest.size);
    INFO_LOG("runMode: %d", runMode);
    if (runMode == ACL_HOST)
    {
        ret = aclrtMemcpy(data_buffer_dev, dest.size, YUV_data.data, dest.size, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    else
    {
        ret = aclrtMemcpy(data_buffer_dev, dest.size, YUV_data.data, dest.size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }

    if (ret != ACL_ERROR_NONE)
    {
        ERROR_LOG("memcpy failed. Input host buffer size is %u", dest.size);
        acldvppFree(data_buffer_dev);
        return FAILED;
    }

    dest.device = KL_DEVICE;
    dest.format = KL_YUV420SP_NV12;
    dest.data = SHARED_PRT_DVPP_BUF(data_buffer_dev);

    return SUCCESS;
}

Result Utils::CvtYuv420spToMat(KLImageData &src, cv::Mat &dest)
{
    aclrtRunMode runMode;
    aclrtGetRunMode(&runMode);

    if (runMode == ACL_HOST)
    {
        void *hostPtr = nullptr;
        aclrtMallocHost(&hostPtr, src.size);
        aclrtMemcpy(hostPtr, src.size, src.data.get(), src.size, ACL_MEMCPY_DEVICE_TO_HOST);
        cv::Mat out_yuv_mat(cv::Size(src.align_width, src.align_height * 3 / 2), CV_8UC1, hostPtr);
        cv::cvtColor(out_yuv_mat, dest, cv::COLOR_YUV2BGR_NV12);
        (void)aclrtFreeHost(hostPtr);
    }
    else
    {
        cv::Mat out_yuv_mat(cv::Size(src.align_width, src.align_height * 3 / 2), CV_8UC1, src.data.get());
        cv::cvtColor(out_yuv_mat, dest, cv::COLOR_YUV2BGR_NV12);
    }
    return SUCCESS;
}

Result Utils::SaveDvppOutputData(const char *fileName, const void *devPtr, uint32_t dataSize)
{
    FILE * outFileFp = fopen(fileName, "wb+");
    aclrtRunMode runMode;
    aclrtGetRunMode(&runMode);
    if (runMode == ACL_HOST) {
        void* hostPtr = nullptr;
        aclrtMallocHost(&hostPtr, dataSize);
        aclrtMemcpy(hostPtr, dataSize, devPtr, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        fwrite(hostPtr, sizeof(char), dataSize, outFileFp);
        (void)aclrtFreeHost(hostPtr);
    }
    else{
        fwrite(devPtr, sizeof(char), dataSize, outFileFp);
    }
    fflush(outFileFp);
    fclose(outFileFp);

    return SUCCESS;
}

void *Utils::CopyDataDeviceToLocal(void *deviceData, uint32_t dataSize)
{
    uint8_t *buffer = new uint8_t[dataSize];
    if (buffer == nullptr)
    {
        ERROR_LOG("New malloc memory failed");
        return nullptr;
    }

    aclError aclRet = aclrtMemcpy(buffer, dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE)
    {
        ERROR_LOG("Copy device data to local failed, aclRet is %d", aclRet);
        delete[](buffer);
        return nullptr;
    }

    return (void *)buffer;
}

void *Utils::CopyDataToDevice(void *data, uint32_t dataSize, aclrtMemcpyKind policy)
{
    void *buffer = nullptr;
    aclError aclRet = aclrtMalloc(&buffer, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_ERROR_NONE)
    {
        ERROR_LOG("malloc device data buffer failed, aclRet is %d", aclRet);
        return nullptr;
    }

    aclRet = aclrtMemcpy(buffer, dataSize, data, dataSize, policy);
    if (aclRet != ACL_ERROR_NONE)
    {
        ERROR_LOG("Copy data to device failed, aclRet is %d", aclRet);
        (void)aclrtFree(buffer);
        return nullptr;
    }

    return buffer;
}

void *Utils::CopyDataDeviceToDevice(void *deviceData, uint32_t dataSize)
{
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
}

void *Utils::CopyDataHostToDevice(void *deviceData, uint32_t dataSize)
{
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
}