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

* File dvpp_process.cpp
* Description: handle dvpp process
*/

#include <iostream>
#include "acl/acl.h"
#include "dvpp_cropandpaste.h"

using namespace std;

DvppCropAndPaste::DvppCropAndPaste(aclrtStream &stream, const uint32_t &out_width, const uint32_t &out_height)
: stream_(stream), dvppChannelDesc_(nullptr), vpcInputDesc_(nullptr), vpcOutputDesc_(nullptr),
    vpcOutBufferDev_(nullptr), cropArea_(nullptr), pasteArea_(nullptr)
{
    in_size_.width = 0;
    in_size_.height = 0;
    out_size_.width = out_width;
    out_size_.height = out_height;

    input_format_ = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
}

DvppCropAndPaste::~DvppCropAndPaste() {
    DestroyCropAndPasteResource();
}

Result DvppCropAndPaste::InitCropAndPasteInputDesc()
{
    uint32_t alignWidth = ALIGN_UP16(in_size_.width);
    uint32_t alignHeight = ALIGN_UP2(in_size_.height);
    if (alignWidth == 0 || alignHeight == 0) {
        ERROR_LOG("Invalid image parameters, width %d, height %d",
                        in_size_.width, in_size_.height);
        return FAILED;
    }

    vpcInBufferSize_  = YUV420SP_SIZE(alignWidth, alignHeight);
    if (vpcInputDesc_ == nullptr) {
        vpcInputDesc_ = acldvppCreatePicDesc();
        if (vpcInputDesc_ == nullptr) {
            ERROR_LOG("acldvppCreatePicDesc vpcInputDesc_ failed");
            return FAILED;
        }
    }
    
    // acldvppSetPicDescData(vpcInputDesc_, inputImage.data.get());
    acldvppSetPicDescFormat(vpcInputDesc_, input_format_);
    acldvppSetPicDescWidth(vpcInputDesc_, in_size_.width);
    acldvppSetPicDescHeight(vpcInputDesc_, in_size_.height);
    acldvppSetPicDescWidthStride(vpcInputDesc_, alignWidth);
    acldvppSetPicDescHeightStride(vpcInputDesc_, alignHeight);
    acldvppSetPicDescSize(vpcInputDesc_, vpcInBufferSize_);

    // must even
    uint32_t cropLeftOffset = 0;
    // must even
    uint32_t cropTopOffset = 0;
    int crop_width = in_size_.width;
    int crop_height = in_size_.height;

    // must odd
    uint32_t cropRightOffset = crop_width % 2 != 0 ? crop_width : crop_width - 1;
    // must odd
    uint32_t cropBottomOffset = crop_height % 2 != 0 ? crop_height : crop_height - 1;

    if (cropArea_ == nullptr)
    {
        cropArea_ = acldvppCreateRoiConfig(cropLeftOffset, cropRightOffset,
                                            cropTopOffset, cropBottomOffset);
        if (cropArea_ == nullptr) {
            ERROR_LOG("acldvppCreateRoiConfig cropArea_ failed");
            return FAILED;
        }
    }

    auto ret = acldvppSetRoiConfig(cropArea_, cropLeftOffset, cropRightOffset, cropTopOffset, cropBottomOffset);
    if (ret != ACL_ERROR_NONE)
    {
        ERROR_LOG("Set input RoiConfig failed");
        return FAILED;
    }

    return SUCCESS;
}

Result DvppCropAndPaste::InitCropAndPasteOutputDesc()
{
    int resizeOutWidth = out_size_.width;
    int resizeOutHeight = out_size_.height;
    int resizeOutWidthStride = ALIGN_UP16(resizeOutWidth);
    int resizeOutHeightStride = ALIGN_UP2(resizeOutHeight);

    if (resizeOutWidthStride == 0 || resizeOutHeightStride == 0) {
        ERROR_LOG("InitResizeOutputDesc AlignmentHelper failed");
        return FAILED;
    }

    vpcOutBufferSize_ = YUV420SP_SIZE(resizeOutWidthStride, resizeOutHeightStride);

    aclError aclRet = acldvppMalloc(&vpcOutBufferDev_, vpcOutBufferSize_);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("acldvppMalloc vpcOutBufferDev_ failed, aclRet = %d", aclRet);
        return FAILED;
    }

    vpcOutputDesc_ = acldvppCreatePicDesc();
    if (vpcOutputDesc_ == nullptr) {
        ERROR_LOG("acldvppCreatePicDesc vpcOutputDesc_ failed");
        return FAILED;
    }
    acldvppSetPicDescData(vpcOutputDesc_, vpcOutBufferDev_);
    acldvppSetPicDescFormat(vpcOutputDesc_, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(vpcOutputDesc_, resizeOutWidth);
    acldvppSetPicDescHeight(vpcOutputDesc_, resizeOutHeight);
    acldvppSetPicDescWidthStride(vpcOutputDesc_, resizeOutWidthStride);
    acldvppSetPicDescHeightStride(vpcOutputDesc_, resizeOutHeightStride);
    acldvppSetPicDescSize(vpcOutputDesc_, vpcOutBufferSize_);

    return SUCCESS;
}

Result DvppCropAndPaste::InitResource()
{
    if (dvppChannelDesc_ == nullptr)
    {
        dvppChannelDesc_ = acldvppCreateChannelDesc();
        if (dvppChannelDesc_ == nullptr)
        {
            ERROR_LOG("acldvppCreateChannelDesc failed");
            return FAILED;
        }

        auto aclRet = acldvppCreateChannel(dvppChannelDesc_);
        if (aclRet != ACL_ERROR_NONE)
        {
            ERROR_LOG("acldvppCreateChannel failed, aclRet = %d", aclRet);
            return FAILED;
        }
    }

    if (SUCCESS != InitCropAndPasteOutputDesc()) {
        ERROR_LOG("InitCropAndPasteOutputDesc failed");
        return FAILED;
    }
    
    return SUCCESS;
}

Result DvppCropAndPaste::SetInputSize(const uint32_t &in_width, const uint32_t &in_height)
{
    if (in_width == in_size_.width && in_height == in_size_.height)
        return SUCCESS;
    
    in_size_.width = in_width;
    in_size_.height = in_height;

    if (SUCCESS != InitCropAndPasteInputDesc()) {
        ERROR_LOG("InitCropAndPasteInputDesc failed");
        return FAILED;
    }

    return SUCCESS;
}

Result DvppCropAndPaste::SetPasteRoi(const CropRect &targetPasteRect)
{
    targetPasteRect_ = targetPasteRect;
    // must even
    uint32_t pasteLeftOffset = targetPasteRect_.x;
    // must even
    uint32_t pasteTopOffset = targetPasteRect_.y;
    auto crop_width = targetPasteRect_.x + targetPasteRect_.width;
    auto crop_height = targetPasteRect_.y + targetPasteRect_.height;
    // must odd
    uint32_t pasteRightOffset = crop_width % 2 != 0 ? crop_width : crop_width - 1;
    // must odd
    uint32_t pasteBottomOffset = crop_height % 2 != 0 ? crop_height : crop_height - 1;

    if (pasteArea_ == nullptr)
    {
        pasteArea_ = acldvppCreateRoiConfig(pasteLeftOffset, pasteRightOffset,
                                                pasteTopOffset, pasteBottomOffset);
        if (pasteArea_ == nullptr) {
            ERROR_LOG("acldvppCreateRoiConfig pasteArea_ failed");
            return FAILED;
        }
    }
    acldvppSetRoiConfig(pasteArea_, pasteLeftOffset, pasteRightOffset, pasteTopOffset, pasteBottomOffset);
    
    return SUCCESS;
}

Result DvppCropAndPaste::CropAndPasteProcess(const KLImageData& srcImage, KLImageData& resizedImage)
{
    // check input
    if (srcImage.device == KL_DEVICE && srcImage.format == KL_YUV420SP_NV12)
    {
        assert(srcImage.width == in_size_.width && srcImage.height == in_size_.height);
    }
    else
    {
        return FAILED;
    }

    if (!cropArea_ || !pasteArea_)
    {
        ERROR_LOG("src input size or targetPasteRect is not set");
        return FAILED;
    }

    auto ret = acldvppSetPicDescData(vpcInputDesc_, srcImage.data.get());
    if (ret != ACL_ERROR_NONE)
    {
        ERROR_LOG("Set input PicDescData failed");
        return FAILED;
    }

    aclError aclRet = acldvppVpcCropAndPasteAsync(dvppChannelDesc_, vpcInputDesc_,
    vpcOutputDesc_, cropArea_, pasteArea_, stream_);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("acldvppVpcCropAndPasteAsync failed, aclRet = %d", aclRet);
        return FAILED;
    }

    aclRet = aclrtSynchronizeStream(stream_);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("crop and paste aclrtSynchronizeStream failed, aclRet = %d", aclRet);
        return FAILED;
    }

    void* tmpOutBufferDev = nullptr;
    aclRet = acldvppMalloc(&tmpOutBufferDev, vpcOutBufferSize_);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("acldvppMalloc vpcOutBufferDev_ failed, aclRet = %d", aclRet);
        return FAILED;
    }
    aclRet = aclrtMemcpy(tmpOutBufferDev, vpcOutBufferSize_,
                            vpcOutBufferDev_, vpcOutBufferSize_, ACL_MEMCPY_DEVICE_TO_DEVICE);

    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("aclrtMemcpy vpcOutBufferDev_ failed, aclRet = %d", aclRet);
        return FAILED;
    }

    aclrtMemset(vpcOutBufferDev_, vpcOutBufferSize_, 0, vpcOutBufferSize_);

    resizedImage.width = out_size_.width;
    resizedImage.height = out_size_.height;
    resizedImage.align_width = ALIGN_UP16(out_size_.width);
    resizedImage.align_height = ALIGN_UP2(out_size_.height);
    resizedImage.size = vpcOutBufferSize_;
    resizedImage.data = SHARED_PRT_DVPP_BUF(tmpOutBufferDev);
    resizedImage.format = KL_YUV420SP_NV12;

    return SUCCESS;
}

void DvppCropAndPaste::DestroyCropAndPasteResource()
{
    if (cropArea_ != nullptr) {
        (void)acldvppDestroyRoiConfig(cropArea_);
        cropArea_ = nullptr;
    }

    if (pasteArea_ != nullptr) {
        (void)acldvppDestroyRoiConfig(pasteArea_);
        pasteArea_ = nullptr;
    }

    if (vpcInputDesc_ != nullptr) {
        (void)acldvppDestroyPicDesc(vpcInputDesc_);
        vpcInputDesc_ = nullptr;
    }

    if (vpcOutputDesc_ != nullptr) {
        (void)acldvppDestroyPicDesc(vpcOutputDesc_);
        vpcOutputDesc_ = nullptr;
    }

    if (dvppChannelDesc_ != nullptr)
    {
        auto aclRet = acldvppDestroyChannel(dvppChannelDesc_);
        if (aclRet != ACL_ERROR_NONE)
        {
            ERROR_LOG("acldvppDestroyChannel failed, aclRet = %d", aclRet);
        }

        (void)acldvppDestroyChannelDesc(dvppChannelDesc_);
        dvppChannelDesc_ = nullptr;
    }
}
