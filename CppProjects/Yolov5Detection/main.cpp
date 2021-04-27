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

* File main.cpp
* Description: dvpp sample main func
*/

#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <vector>
#include <chrono>

// #include "object_detect_yolov3.h"
#include "object_detect_yolov5_bgr.h"
#include "path_operate.h"

// #include "dvpp_resize_test.h"
// #include "dvpp_cropandpaste_test.h"


using namespace std;


const static std::vector<std::string> yolov3Label = {"person", "bicycle", "car", "motorbike",
                                                        "aeroplane", "bus", "train", "truck", "boat",
                                                        "traffic light", "fire hydrant", "stop sign", "parking meter",
                                                        "bench", "bird", "cat", "dog", "horse",
                                                        "sheep", "cow", "elephant", "bear", "zebra",
                                                        "giraffe", "backpack", "umbrella", "handbag", "tie",
                                                        "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                                                        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                                        "tennis racket", "bottle", "wine glass", "cup",
                                                        "fork", "knife", "spoon", "bowl", "banana",
                                                        "apple", "sandwich", "orange", "broccoli", "carrot",
                                                        "hot dog", "pizza", "donut", "cake", "chair",
                                                        "sofa", "potted plant", "bed", "dining table", "toilet",
                                                        "TV monitor", "laptop", "mouse", "remote", "keyboard",
                                                        "cell phone", "microwave", "oven", "toaster", "sink",
                                                        "refrigerator", "book", "clock", "vase", "scissors",
                                                        "teddy bear", "hair drier", "toothbrush"};


const static std::vector<std::string> litterLabel = {"litter", "litter prob", "lid bin close", "lid bin open", "unlid bin", "spilled bin"};
const static std::vector<std::string> carLabel = {"car", "truck", "bicycle", "motorcycle", "bus", "tricycle"};
const static std::vector<std::string> hardhatLabel = {"hard hat","head","Head gear","red hat","white peaked cap","chef hat","bar chef hat"};
const static std::vector<std::string> personcarLabel = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "tricycle1", "tricycle2", "tricycle3"};
const static std::vector<std::string> sloganLabel = {"slogan", "billboard"};
const static std::vector<std::string> manholdLabel = {"manhold_cover", "manhold_cover_break", "manhold_cover_lost"};
const static std::vector<std::string> dustLabel = {"bicycle","bus","car","motorcycle","person","tricycle","tricycle1","tricycle2","tricycle3","truck",
                                                   "excavator","hoist","crane","cutting","dust","havedust", "dust1"};
const static std::vector<std::string> fireSmokeLabel = {"fire","smoke"};
const static std::vector<std::string> trushLabel = {"litter", "litter prob", "lid bin close", "lid bin open", "unlid bin", "spilled bin"};
const static std::vector<std::string> potholeLabel = {"pothole"};
const static std::vector<std::string> muckLabel = {"muck"};
const static std::vector<std::string> plateLabel = {"blue plate", "yellow plate", "white plate", "green plate", "double-yellow plate","double-white plate", 
                                                    "double-green plate", "double-blue plate"};
const static std::vector<std::string> soilLabel = {"crack", "breakage", "bare soil"};
const static std::vector<std::string> stallLabel = {"stool","desk","canopy","stall0","stall1"};
const static std::vector<std::string> waterLabel = {"water"};
// const static std::vector<std::string> waterDetLabel = {"water"};



Result PadResize(const cv::Mat &src_image, const int &dest_width, const int &dest_height, cv::Mat &dest_image, float &scale)
{
    int image_w = src_image.cols;
    int image_h = src_image.rows;
    scale = 1.f;
    dest_image = cv::Mat(cv::Size(dest_width, dest_height), CV_8UC3, cv::Scalar(114, 114, 114));
    // bool is_width_long = image_w > image_h ? true : false;
    cv::Size target_size;
    if (image_w > dest_width && image_h < dest_height)
    {
        scale = image_w * 1.f / dest_width;
        target_size = cv::Size(dest_width, std::round(image_h / scale));
    }
    else if (image_w < dest_width && image_h > dest_height)
    {
        scale = image_h * 1.f / dest_height;
        target_size = cv::Size(std::round(image_w / scale), dest_height);
    }
    else if (image_w > dest_width && image_h > dest_height)
    {
        if (image_w * 1.f / dest_width > image_h * 1.f / dest_height)
        {
            scale = image_w * 1.f / dest_width;
            target_size = cv::Size(dest_width, std::round(image_h / scale));
        }
        else
        {
            scale = image_h * 1.f / dest_height;
            target_size = cv::Size(std::round(image_w / scale), dest_height);
        }
    }
    else
    {
        // patch to model input size
        cv::Mat input_roi = dest_image(cv::Rect(0, 0, image_w, image_h));
        src_image.copyTo(input_roi);

        return SUCCESS;
    }

    cv::Mat resized_mat;
    cv::resize(src_image, resized_mat, target_size);

    // patch to model input size
    cv::Mat input_roi = dest_image(cv::Rect(0, 0, target_size.width, target_size.height));
    resized_mat.copyTo(input_roi);

    return SUCCESS;
}

int test_yolov5_rgb(std::shared_ptr<ObjectDetectYolov5BGR> yolov5_ptr,
                    std::string image_path = "../data/hardhat.jpeg",
                    std::vector<std::string> label_txt = hardhatLabel,
                    int repeat = 1
                    )
{
    INFO_LOG("Test image: %s", image_path.c_str());
    fs::path file_path(image_path);
    fs::path out_dir = file_path.parent_path()/"out";

    if (!fs::exists(out_dir))
        fs::create_directory(out_dir);
    
    fs::path save_path = out_dir/file_path.filename();

    // ObjectDetectYolov5BGR yolov5(model_path.c_str(), is_use_aipp, 1, 416, 416);
    // auto ret = yolov5.Init();
    // if (ret != SUCCESS)
    // {
    //     ERROR_LOG("yolov3 Init failed");
    //     return -1;
    // }

    cv::Mat cv_image = cv::imread(image_path);
    cv::Mat pad_1080p;
    // float pad_scale;
    // PadResize(cv_image, 1920, 1080, pad_1080p, pad_scale);
    pad_1080p = cv_image;
    
    std::vector<BBox> bboxes;

    // // warmup 5 loop
    // for (int i = 0; i < 5; i ++)
    // {
    //     ret = yolov5.Preprocess(pad_1080p);
    //     if (ret != SUCCESS)
    //     {
    //         ERROR_LOG("yolov3 Preprocess failed");
    //         return -1;
    //     }
        
    //     ret = yolov5.Inference(bboxes);
    //     if (ret != SUCCESS)
    //     {
    //         ERROR_LOG("yolov3 Inference failed");
    //         return -1;
    //     }
    //     INFO_LOG("yolov5 Inference done");
    // }
    // INFO_LOG("Warmup Done!");

    auto start_t = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < repeat; i ++)
    {
        auto ret = yolov5_ptr->Preprocess(pad_1080p);
        if (ret != SUCCESS)
        {
            ERROR_LOG("yolov5 Preprocess failed");
            return -1;
        }
        ret = yolov5_ptr->Inference(bboxes);
        if (ret != SUCCESS)
        {
            ERROR_LOG("yolov5 Inference failed");
            return -1;
        }
        INFO_LOG("yolov5 Inference done");
    }
    
    auto end_t = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t);
    INFO_LOG("Yolov5 Inference cost time: %.2f ms", float(elapsed_ms.count() / repeat));

    INFO_LOG("Detect the number of object is: %zu", bboxes.size());
    for (size_t i = 0; i < bboxes.size(); i ++)
    {
        const auto &rect = bboxes[i].rect;
        const auto &cls_id = bboxes[i].cls_id;
        const auto &conf = bboxes[i].score;
        // if (cls_id == 1)
        //     continue;

        if (conf < 0.4)
            continue;
        cv::rectangle(pad_1080p, rect, cv::Scalar(0, 0, 255), 2);
        std::string text = label_txt[cls_id] + "_" + std::to_string(conf);
        // std::string text = std::to_string(cls_id) + "_" + std::to_string(conf);

        cv::putText(pad_1080p, text, cv::Point2i(rect.x + 5, rect.y + 15), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
        // INFO_LOG("%s, location [%d, %d, %d, %d]", text.c_str(), rect.x, rect.y, rect.width, rect.height);
    }

    cv::imwrite(save_path.string(), pad_1080p);
    INFO_LOG("Save out image: %s", save_path.string().c_str());

    return 0;
}

int TestYolov5Yuv(std::shared_ptr<ObjectDetectYolov5BGR> &yolov5_ptr, 
                    const std::string &image_path, 
                    const std::vector<std::string> &label_txt,
                    const int repeat = 1)
{
    INFO_LOG("Test image: %s", image_path.c_str());
    fs::path file_path(image_path);
    fs::path out_dir = file_path.parent_path()/"out";

    if (!fs::exists(out_dir))
        fs::create_directory(out_dir);
    
    fs::path save_path = out_dir/file_path.filename();

    cv::Mat cv_show = cv::imread(image_path);

    KLImageData input_image;
    Utils::CvtMatToYuv420sp(cv_show, input_image);

    std::vector<BBox> bboxes;
    auto start_t = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < repeat; i ++)
    {
        INFO_LOG("yolov5 Preprocess");
        auto ret = yolov5_ptr->Preprocess(input_image);
        if (ret != SUCCESS)
        {
            ERROR_LOG("yolov5 Preprocess failed");
            return -1;
        }

        INFO_LOG("yolov5 inference");
        ret = yolov5_ptr->Inference(bboxes);
        if (ret != SUCCESS)
        {
            ERROR_LOG("yolov5 Inference failed");
            return -1;
        }
        INFO_LOG("yolov5 Inference done");
    }
    
    auto end_t = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t);
    INFO_LOG("Yolov5 Inference cost time: %.2f ms", float(elapsed_ms.count() / repeat));

    INFO_LOG("Detect the number of object is: %zu", bboxes.size());
    if (bboxes.empty())
    {
        fs::path out_dir = file_path.parent_path()/"out_empty";

        if (!fs::exists(out_dir))
            fs::create_directory(out_dir);
        
        fs::path save_path = out_dir/file_path.filename();
        cv::imwrite(save_path.string(), cv_show);
        INFO_LOG("Save out image: %s", save_path.string().c_str());
        return 0;
    }

    for (size_t i = 0; i < bboxes.size(); i ++)
    {
        const auto &rect = bboxes[i].rect;
        const auto &cls_id = bboxes[i].cls_id;
        const auto &conf = bboxes[i].score;
        // if (cls_id == 1)
        //     continue;

        if (conf < 0.4)
            continue;
        cv::rectangle(cv_show, rect, cv::Scalar(0, 0, 255), 2);
        std::string text = label_txt[cls_id] + "_" + std::to_string(conf);
        // std::string text = std::to_string(cls_id) + "_" + std::to_string(conf);

        cv::putText(cv_show, text, cv::Point2i(rect.x + 5, rect.y + 15), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
        // INFO_LOG("%s, location [%d, %d, %d, %d]", text.c_str(), rect.x, rect.y, rect.width, rect.height);
    }

    cv::imwrite(save_path.string(), cv_show);
    INFO_LOG("Save out image: %s", save_path.string().c_str());

    return 0;
}

int TestCropAndPaste(std::shared_ptr<DvppCropAndPaste> &dvpp_processor, const std::string &file_name)
{
    INFO_LOG("Test image: %s", file_name.c_str());
    fs::path file_path(file_name);
    fs::path out_dir = file_path.parent_path()/"out_inputs";

    if (!fs::exists(out_dir))
        fs::create_directory(out_dir);
    
    // fs::path save_path = out_dir/file_path.filename();

    cv::Mat cv_show = cv::imread(file_name);

    KLImageData input_image, resized_image;
    Utils::CvtMatToYuv420sp(cv_show, input_image);

    // std::shared_ptr<DvppCropAndPaste> dvpp_processor = std::make_shared<DvppCropAndPaste>(stream, 768, 768);
    dvpp_processor->SetInputSize(input_image.width, input_image.height);

    CropRect paste_rect;
    float scale;
    Utils::PadResize(input_image.width, input_image.height, 768, 768, paste_rect, scale);
    dvpp_processor->SetPasteRoi(paste_rect);

    dvpp_processor->CropAndPasteProcess(input_image, resized_image);
    std::string out_file_name = file_path.filename().stem().string() + ".yuv";
    fs::path save_path = out_dir/out_file_name;
    Utils::SaveDvppOutputData(save_path.c_str(), resized_image.data.get(), resized_image.size);

    return 0;
}

int TestCropAndPasteDir()
{
    std::string image_dir = "../data/person_car/";
    std::vector<std::string> image_files;
    std::vector<std::string> jpg_files = Path::getfiles(image_dir, ".jpg");
    image_files.insert(image_files.end(), jpg_files.begin(), jpg_files.end());
    std::vector<std::string> png_files = Path::getfiles(image_dir, ".png");
    image_files.insert(image_files.end(), png_files.begin(), png_files.end());
    std::vector<std::string> jpeg_files = Path::getfiles(image_dir, ".jpeg");
    image_files.insert(image_files.end(), jpeg_files.begin(), jpeg_files.end());


    aclInit(nullptr);
    INFO_LOG("acl init success");
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;

    /* 2.Run the management resource application, including Device, Context, Stream */
    aclrtSetDevice(deviceId);
    aclrtCreateContext(&context, deviceId);
    aclrtCreateStream(&stream);

    std::shared_ptr<DvppCropAndPaste> dvpp_processor = std::make_shared<DvppCropAndPaste>(stream, 768, 768);
    dvpp_processor->InitResource();

    for (const auto &file_name : image_files)
    {
        std::string image_path = image_dir + "/" + file_name;
        TestCropAndPaste(dvpp_processor, image_path);
    }

    aclError ret;
    if (stream != nullptr) {
        ret = aclrtDestroyStream(stream);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context != nullptr) {
        ret = aclrtDestroyContext(context);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = aclrtResetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");


    return 0;
}

int test_dir()
{
    std::string image_dir = "../data/person_car/";
    std::vector<std::string> image_files;
    std::vector<std::string> jpg_files = Path::getfiles(image_dir, ".jpg");
    image_files.insert(image_files.end(), jpg_files.begin(), jpg_files.end());
    std::vector<std::string> png_files = Path::getfiles(image_dir, ".png");
    image_files.insert(image_files.end(), png_files.begin(), png_files.end());
    std::vector<std::string> jpeg_files = Path::getfiles(image_dir, ".jpeg");
    image_files.insert(image_files.end(), jpeg_files.begin(), jpeg_files.end());

    INFO_LOG("=======>>> file num: %zu", image_files.size());

    std::string model_path = "../models/person_car_768_yuv-20210416.om";
    std::shared_ptr<ObjectDetectYolov5BGR> yolov5_ptr = std::make_shared<ObjectDetectYolov5BGR>(model_path.c_str(), true, 0, 768, 768);
    auto ret = yolov5_ptr->Init();
    if (ret != SUCCESS)
    {
        ERROR_LOG("yolov5 Init failed");
        return -1;
    }

    for (const auto &file_name : image_files)
    {
        std::string image_path = image_dir + "/" + file_name;
        // test_yolov5_rgb(yolov5_ptr, image_path, personcarLabel);
        TestYolov5Yuv(yolov5_ptr, image_path, personcarLabel);
    }

    return 0;
}

int main(int argc, char* argv[])
{

    // test yolov5 om
    // test_yolov5_rgb();

    test_dir();

    // TestCropAndPasteDir();

    // assert(argc = 4);
    // std::string yuv_file_name = argv[1];
    // int width = std::atoi(argv[2]);
    // int height = std::atoi(argv[3]);
    // DvppResizeTest(yuv_file_name, width, height);
    // DvppCropAndPasteTest(yuv_file_name, width, height);

    return 0;
}
