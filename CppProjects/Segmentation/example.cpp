#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include "utils.h"
// #include "segmentation.h"
// #include "utils.h"
#include "segmentation_infer.h"
using namespace std;
using namespace atlas_segmt_dlv3;


int TestDLV3()
{
    // std::string model_path = "../models/DeepLabV3_framework_tensorflow_ascend310_input_uint8_batch_1_fp16_output_fp32.om";    //使用与下面om模型对应的pb模型自行转出来的模型
    std::string model_path = "../models/deeplabv3_framework_tensorflow_aipp_0_batch_1_input_fp16_output_FP32.om";
    ISegmentation processSample(model_path, 0, 500, 375);
    
    string input_path = "../data/002-01500.jpg";
    cv::Mat image = cv::imread(input_path);
    cv::Mat image_mask;
    image_mask = processSample.Excute(image);

    cv::imwrite("./boat_mask.png", image_mask);

    cv::Mat color_pre;
    Utils::DrawMask(color_pre, image, image_mask);
    cv::imwrite("./boat_color.jpg", color_pre);

    return 0;
}

int TestCityV3()
{
    std::string model_path = "../models/deeplabv3_city_input_NHWC_uint8_512_512_3_output_FP32.om";    //使用与下面om模型对应的pb模型自行转出来的模型
    // std::string model_path = "../models/deeplabv3_framework_tensorflow_aipp_0_batch_1_input_fp16_output_FP32.om";
    ISegmentation processSample(model_path, 0, 512, 512);
    
    string input_path = "../data/002-01500.jpg";
    cv::Mat image = cv::imread(input_path);
    cv::Mat image_mask;
    image_mask = processSample.Excute(image);

    cv::imwrite("./city_mask.png", image_mask);

    cv::Mat color_pre;
    Utils::DrawMask(color_pre, image, image_mask);
    cv::imwrite("./city_color.jpg", color_pre);

    return 0;
}


int main(int argc, char *argv[])
{
    TestDLV3();
    // TestCityV3();
    // TestMat();
    return 0;
}