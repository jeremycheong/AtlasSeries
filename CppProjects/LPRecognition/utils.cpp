#include "utils.h"

Result Utils::PadResize(const cv::Mat& src_image, const int& dest_long_size, cv::Mat& dest_image, float& scale)
{
    int long_size = src_image.cols > src_image.rows ? src_image.cols : src_image.rows;
    bool is_long_width = src_image.cols > src_image.rows ? true : false;
    scale = long_size * 1.0f / dest_long_size;
    int dest_width = 0;
    int dest_height = 0;
    if (is_long_width)
    {
        dest_width = dest_long_size;
        dest_height = src_image.rows / scale;
    }
    else
    {
        dest_width = src_image.cols / scale;
        dest_height = dest_long_size;
    }

    cv::Mat resized;
    cv::resize(src_image, resized, cv::Size(dest_width, dest_height));
    dest_image = cv::Mat(cv::Size(dest_long_size, dest_long_size), CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(dest_image(cv::Rect(0, 0, dest_width, dest_height)));

    return SUCCESS;
}

Result Utils::Focus(const std::vector<cv::Mat>& rgb_channels, std::vector<cv::Mat>& splited_channels)
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
    for (size_t ch = 0; ch < rgb_channels.size(); ch++)
    {
        const cv::Mat& ch_mat = rgb_channels[ch];

        cv::Mat dest_mat00(dest_width, dest_height, CV_32FC1);
        cv::Mat dest_mat01(dest_width, dest_height, CV_32FC1);
        cv::Mat dest_mat10(dest_width, dest_height, CV_32FC1);
        cv::Mat dest_mat11(dest_width, dest_height, CV_32FC1);
        for (int row = 0; row < dest_height; row++)
        {
            for (int col = 0; col < dest_width; col++)
            {
                dest_mat00.at<float>(row, col) = ch_mat.at<float>(2 * row + 0, 2 * col + 0);
                dest_mat01.at<float>(row, col) = ch_mat.at<float>(2 * row + 0, 2 * col + 1);
                dest_mat10.at<float>(row, col) = ch_mat.at<float>(2 * row + 1, 2 * col + 0);
                dest_mat11.at<float>(row, col) = ch_mat.at<float>(2 * row + 1, 2 * col + 1);
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

float Utils::Sigmoid(const float& val)
{
    return 1.0 / (1.0 + exp(-val));
}

std::vector<float> Utils::Sigmoid_1D(const std::vector<float> &data)
{
    std::vector<float> results;
    for (const auto &ele : data)
    {
        results.emplace_back(1.0 / (1.0 + exp(-ele)));
    }
    return results;
}

int Utils::ArgMax_1D(const std::vector<float>& data)
{
    auto max_ele_iter = std::max_element(data.begin(), data.end());
    return std::distance(data.begin(), max_ele_iter);
}

std::vector<float> Utils::Softmax_1D(const std::vector<float>& data)
{
    std::vector<float> results;
    auto max_iter = std::max_element(data.begin(), data.end());
    float ele_sum = 0.f;
    for (size_t i = 0; i < data.size(); i ++)
    {
        float ele_exp = std::exp(data[i] - *max_iter);
        // float ele_exp = std::exp2f(data[i]);
        ele_sum += ele_exp;
        results.emplace_back(ele_exp);
    }

    for (auto &res : results)
    {
        res /= ele_sum;
    }

    return results;
}

void Utils::Softmax_2D_inplace(float* data_ptr, const int& height, const int& width, const int& dim)
{
    if (dim == 0)
    {
        std::vector<float> max_vals(width, -FLT_MAX);
        for (int row = 0; row < height; row++)
        {
            const float* row_data_ptr = data_ptr + row * width;
            for (int col = 0; col < width; col++)
            {
                max_vals[col] = std::max(max_vals[col], row_data_ptr[col]);
            }
        }
        std::vector<float> sum_vals(width, 0.f);
        for (int row = 0; row < height; row++)
        {
            float* row_data_ptr = data_ptr + row * width;
            for (int col = 0; col < width; col++)
            {
                row_data_ptr[col] = static_cast<float>(exp(row_data_ptr[col] - max_vals[col]));
                sum_vals[col] += row_data_ptr[col];
            }
        }

        for (int row = 0; row < height; row++)
        {
            float* row_data_ptr = data_ptr + row * width;
            for (int col = 0; col < width; col++)
            {
                row_data_ptr[col] /= sum_vals[col];
            }
        }
        return;
    }
    else if (dim == 1)
    {
        for (int row = 0; row < height; row++)
        {
            float* row_data_ptr = data_ptr + row * width;
            float max_val = -FLT_MAX;
            for (int col = 0; col < width; col++)
            {
                max_val = std::max(max_val, row_data_ptr[col]);
            }
            float sum_val = 0.f;
            for (int col = 0; col < width; col++)
            {
                row_data_ptr[col] = static_cast<float>(exp(row_data_ptr[col] - max_val));
                sum_val += row_data_ptr[col];
            }

            for (int col = 0; col < width; col++)
            {
                row_data_ptr[col] /= sum_val;
            }
        }
        return;
    }
    else
    {
        ERROR_LOG("input dim[%d] must less than 2", dim);
        return;
    }
}

void Utils::QsortDescentInplace(std::vector<BBox>& obj_boxes, int left, int right)
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

void Utils::QsortDescentInplace(std::vector<BBox>& obj_boxes)
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

void Utils::NmsSortedBboxes(const std::vector<BBox>& obj_boxes, std::vector<int>& picked, float nms_threshold)
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

void* Utils::CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize)
{
    uint8_t* buffer = new uint8_t[dataSize];
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

    return (void*)buffer;
}

void* Utils::CopyDataToDevice(void* data, uint32_t dataSize, aclrtMemcpyKind policy)
{
    void* buffer = nullptr;
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

void* Utils::CopyDataDeviceToDevice(void* deviceData, uint32_t dataSize)
{
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
}

void* Utils::CopyDataHostToDevice(void* deviceData, uint32_t dataSize)
{
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
}