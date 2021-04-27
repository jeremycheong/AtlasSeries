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

* File sample_process.h
* Description: handle acl resource
*/
#pragma once
#include "utils.h"
#include "acl/acl.h"
#include <memory>
#include "model_process.h"

template <class Type>
std::shared_ptr<Type> MakeSharedNoThrow()
{
    try
    {
        return std::make_shared<Type>();
    }
    catch (...)
    {
        return nullptr;
    }
}

#define MAKE_SHARED_NO_THROW(memory, memory_type) \
    memory = MakeSharedNoThrow<memory_type>();

namespace atlas_segmt_dlv3
{
    /**
* SampleProcess
*/
    class Segmentation
    {
    public:
        /**
    * @brief Constructor
    */
        Segmentation(const char *modelPath, const size_t &deviceId, const size_t &modelWidth, const size_t &modelHeight);

        /**
    * @brief Destructor
    */
        ~Segmentation();

        /**
    * @brief init reousce
    * @return result
    */
        Result Init();

        Result Inference(const cv::Mat &image, cv::Mat &image_mask);

    private:
        Result InitResource();
        Result InitModel(const char *modelPath);
        Result Preprocess(std::shared_ptr<ImageDesc> &resizedImage, const cv::Mat &image);
        Result PostProcess(cv::Mat &mask, aclmdlDataset *modelOutput, aclmdlDesc *modelDesc);

        void DestroyResource();

        ModelProcess model_;
        size_t model_width_;
        size_t model_height_;

        int input_width_;
        int input_height_;

        const char *model_path_;
        int32_t deviceId_;
        aclrtContext context_;
        aclrtStream stream_;
        bool is_inited_;
        aclrtRunMode runMode_;
    };

} // namespace atlas_segmt_dlv3