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
#include "path_operate.h"

#include "lprecognition.h"

int ProcessResults(const std::string &image_dir, const std::string &out_file_path)
{
    std::ifstream result_file(out_file_path);
    std::string line_content;

    char file_name[64];
    char plr_result[16];
    float result_conf;
    std::ofstream diff_out("./diff_out.txt");
    while (std::getline(result_file, line_content))
    {
        std::sscanf(line_content.data(), "%s %s %f", file_name, plr_result, &result_conf);
        // std::cout << file_name << ", " << plr_result << ", " << result_conf << std::endl;
        std::string file_name_str(file_name);
        std::string plr_result_str(plr_result);
        auto idx_s = file_name_str.find_first_of("_");
        if (idx_s != std::string::npos)
        {
            idx_s += 1;
        }
        else
            idx_s = 0;

        auto idx_e = file_name_str.find_first_of(".");
        if (idx_e == std::string::npos)
        {
            idx_e = file_name_str.size() - 1;
        }

        std::string plate_num;
        plate_num.assign(file_name_str.c_str() + idx_s, file_name_str.c_str() + idx_e);
        if (plate_num != plr_result_str)
        {
            diff_out << line_content << std::endl;
            std::cout << line_content << std::endl;
        }

        std::memset(file_name, 0, sizeof(file_name));
        std::memset(plr_result, 0, sizeof(plr_result));
    }

    diff_out.close();
    
    return 0;
}

std::string TestPLR(std::string image_path,
            const std::shared_ptr<LPRecoginition> &lpr_ptr)
{
    cv::Mat image = cv::imread(image_path);
    // LPRecoginition lpr(refine_model_path, lpr_model_path, BGR_PACKAGE, BGR_PACKAGE);

    // lpr.Init();

    std::string results;
    std::vector<float> result_confs;
    auto ret = lpr_ptr->Inference(image, results, result_confs);
    if (ret != SUCCESS)
    {
        ERROR_LOG("LPR Inference Failed");
        return nullptr;
    }

    float final_conf = 1.0f;
    for (const auto &conf : result_confs)
        final_conf *= conf;

    std::string final_results;
    final_results = results + " " + std::to_string(final_conf);

    return final_results;
}

int TestPLRDir(const std::string &image_dir)
{
    std::string refine_model_path = "../models/LPR_refinenet_withAipp.om";
    std::string lpr_model_path = "../models/LPRNet_high_precision.om";
    std::vector<std::string> filter_ext = {".jpg", ".jpeg"};
    auto image_files = Path::getfiles(image_dir, filter_ext);
    if (image_files.empty())
    {
        ERROR_LOG("Get image files none");
        return -1;
    }

    std::ofstream results_out("./results.txt");
    if (!results_out.is_open())
    {
        ERROR_LOG("open save results file failed");
        return -1;
    }

    std::shared_ptr<LPRecoginition> lpr_ptr = std::make_shared<LPRecoginition>(refine_model_path, lpr_model_path, BGR_PACKAGE, BGR_PACKAGE);
    lpr_ptr->Init();
    for (auto filename : image_files)
    {
        results_out << filename << " ";
        filename = image_dir + "/" + filename;
        std::string result = TestPLR(filename, lpr_ptr);
        results_out << result << std::endl;
    }
    results_out.close();

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage ./LPRecognition image_dir" << std::endl;
        return 0;
    }
    std::string file_path(argv[1]);
    if (fs::is_directory(file_path))
        ProcessResults(file_path, "./results.txt");
        // TestPLRDir(file_path);
    else
        std::cout << "Usage ./LPRecognition image_dir" << std::endl;

    return 0;
}
