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

* File dvpp_process.h
* Description: handle dvpp process
*/
#pragma once
#include <cstdint>

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "utils.h"


class DvppCropAndPaste {
    public:

    /**
     * @brief Construct a new Dvpp Crop And Paste object
     * 
     * @param stream 
     * @param out_width 
     * @param out_height 
     */
    DvppCropAndPaste(aclrtStream &stream, const uint32_t &out_width, const uint32_t &out_height);

    /**
    * @brief Destructor
    */
    ~DvppCropAndPaste();

    /**
    * @brief dvpp global init
    * @return AtlasError
    */
    Result InitResource();

    /**
     * @brief Set the Input Size object
     * 
     * @param in_width 
     * @param in_height 
     * @return AtlasError 
     */
    Result SetInputSize(const uint32_t &in_width, const uint32_t &in_height);

    /**
     * @brief Set the Paste Roi object
     * 
     * @param targetPasteRect 
     * @return AtlasError 
     */
    Result SetPasteRoi(const CropRect &targetPasteRect);

    /**
     * @brief 
     * 
     * @param resizedImage 
     * @param srcImage 
     * @return AtlasError 
     */
    Result CropAndPasteProcess(const KLImageData& srcImage, KLImageData& resizedImage);

    /**
     * @brief 
     * 
     */
    void DestroyCropAndPasteResource();
    
private:
    Result InitCropAndPasteInputDesc();
    Result InitCropAndPasteOutputDesc();


private:
    aclrtStream stream_;
    acldvppChannelDesc *dvppChannelDesc_;

    acldvppPixelFormat input_format_;

    acldvppPicDesc *vpcInputDesc_;
    acldvppPicDesc *vpcOutputDesc_;

    uint32_t vpcInBufferSize_;

    void *vpcOutBufferDev_;
    uint32_t vpcOutBufferSize_;

    acldvppRoiConfig *cropArea_;
    acldvppRoiConfig *pasteArea_;

    Resolution size_;

    CropRect srcCropRect_;
    CropRect targetPasteRect_;
    Resolution in_size_;
    Resolution out_size_;

    bool is_init_;

    uint32_t ltHorz_;
    uint32_t rbHorz_;
    uint32_t ltVert_;
    uint32_t rbVert_;
};

