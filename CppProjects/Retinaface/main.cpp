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

#include "retinaface.h"
#include "path_operate.h"

/*
int TestRetinaface(std::string model_path = "../models/retinaface_withAipp_960x960.om",
                    std::string image_path = "../data/fail/19.jpg",
                    int repeat = 1
                    )
{
    cv::Mat image = cv::imread(image_path);
    FaceDetection fd(model_path);

    fd.Init();

    std::vector<float> face_confs;
    std::vector<cv::Rect> face_rects;
    std::vector<std::vector<cv::Point2f> > face_points;

    for (int i = 0; i < 5; i ++)
    {
        auto ret = fd.Inference(image, face_rects, face_confs, face_points);
        if (ret != SUCCESS)
        {
            ERROR_LOG("LPR Inference Failed");
            return -1;
        }
    }
    INFO_LOG("Warmup Done!");

    auto start_t = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i ++)
    {
        auto ret = fd.Inference(image, face_rects, face_confs, face_points);
        if (ret != SUCCESS)
        {
            ERROR_LOG("LPR Inference Failed");
            return -1;
        }
    }

    auto end_t = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t);
    INFO_LOG("Retinaface Inference cost time: %.2f ms", float(elapsed_ms.count() / repeat));

    INFO_LOG("face_rects size: %zu, face_confs size: %zu", face_rects.size(), face_confs.size());

    for (size_t i = 0; i < face_rects.size(); i ++)
    {
        cv::Rect &face_rect = face_rects[i];
        float face_conf = face_confs[i];
        std::vector<cv::Point2f> &landmk = face_points[i];

        // cv::Rect face_crop_2(int(face_rect.x - face_rect.width / 2), int(face_rect.y - face_rect.height / 2), face_rect.width * 2, face_rect.height * 2);
        // cv::Mat face_crop_2_mat = 

        // INFO_LOG("face_rect shape: [%d, %d, %d, %d]", face_rect.x, face_rect.y, face_rect.width, face_rect.height);
        cv::rectangle(image, face_rect, cv::Scalar(0, 0, 255));
        for (size_t j = 0; j < landmk.size(); j ++)
        {
            // INFO_LOG("landmk[%zu]: (%.4f, %.4f)", j, landmk[j].x, landmk[j].y);
            cv::circle(image, landmk[j], 1, cv::Scalar(0, 0, 255), 2);
        }
        cv::putText(image, std::to_string(face_conf), cv::Point(face_rect.x, face_rect.y -5), 
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 0, 255));
    }

    cv::imwrite("Retinaface_out.jpg", image);

    return 0;
}
*/

void TestRetinaface(const std::shared_ptr<FaceDetection> &retinaface_ptr, const std::string &filename)
{
    std::string out_path = Path::parentpath(filename) + "/out/";
    if (!fs::exists(out_path))
        fs::create_directory(out_path);

    std::string save_path = out_path + Path::splitext(filename)[0] + "_out.jpg";
    INFO_LOG("Save result path: %s", save_path.c_str());
    // std::string model_name = "retinaface";

    cv::Mat img = cv::imread(filename);
    // KLImageData image_data;
    // image_data.format = KL_CV_MAT_BGR888;
    // image_data.width = img.cols;
    // image_data.height = img.rows;
    // image_data.size = img.cols * img.rows * 3;
    // image_data.data.reset(new uint8_t[image_data.size], [](uint8_t* p){delete[](p);});
    // memcpy(image_data.data.get(), img.data, image_data.size);

    std::vector<cv::Rect> rects;
    std::vector<float> confs; 
    std::vector<std::vector<cv::Point2f>> points;

    retinaface_ptr->Inference(img, rects, confs, points);
    // face_detect_atlas(image_data, rects, confs, points, model_name);

    // std::cout << "rects size: " << rects.size() << std::endl;
    INFO_LOG("%s has %zu faces", filename.c_str(), rects.size());

    cv::Rect image_rect(0, 0, img.cols, img.rows);
    for (size_t i = 0; i < rects.size(); i ++)
    {
        cv::Rect &face_rect = rects[i];
        float face_conf = confs[i];
        std::vector<cv::Point2f> &landmk = points[i];

        // std::vector<std::string> result = TestFaceQuality(image_data, landmk);

        cv::rectangle(img, face_rect, cv::Scalar(0, 0, 255));
        for (size_t j = 0; j < landmk.size(); j ++)
        {
            // INFO_LOG("landmk[%zu]: (%.4f, %.4f)", j, landmk[j].x, landmk[j].y);
            cv::circle(img, landmk[j], 1, cv::Scalar(0, 0, 255));
        }
        char text[20];
        // std::string text = std::to_string(face_conf) + " mask: " + result[1];
        sprintf(text, "%.2f", face_conf);
        cv::putText(img, text, cv::Point(face_rect.x, face_rect.y -5), 
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 255));
        // cv::Rect face_2_rect(int(face_rect.x - face_rect.width / 2), int(face_rect.y - face_rect.height / 2), face_rect.width * 2, face_rect.height * 2);

    }

    cv::imwrite(save_path, img);

    std::cout << "============================== TestRetinaface end" << std::endl;

}

void TestRetinaFaceDir()
{
    std::string image_dir = "/home/koala/CityManagerAscend/workspace/mask_test/video";
    std::vector<std::string> image_files;
    std::vector<std::string> jpg_files = Path::getfiles(image_dir, ".jpg");
    image_files.insert(image_files.end(), jpg_files.begin(), jpg_files.end());
    std::vector<std::string> png_files = Path::getfiles(image_dir, ".png");
    image_files.insert(image_files.end(), png_files.begin(), png_files.end());
    std::vector<std::string> jpeg_files = Path::getfiles(image_dir, ".jpeg");
    image_files.insert(image_files.end(), jpeg_files.begin(), jpeg_files.end());

    INFO_LOG("=======>>> file num: %zu", image_files.size());

    std::string model_path = "../models/retinaface_withAipp_960x960.om";
    std::shared_ptr<FaceDetection> retinaface_ptr = std::make_shared<FaceDetection>(model_path.c_str(), 0);
    auto ret = retinaface_ptr->Init();
    if (ret != SUCCESS)
    {
        ERROR_LOG("yolov5 Init failed");
        return;
    }

    for (const auto &file_name : image_files)
    {
        std::string image_path = image_dir + "/" + file_name;
        TestRetinaface(retinaface_ptr, image_path);
    }

    std::string image_path = "";
    TestRetinaface(retinaface_ptr, image_path);

}

int main(int argc, char* argv[])
{
    TestRetinaFaceDir();

    return 0;
}
