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

* File utils.cpp
* Description: handle file operations
*/
#include "utils.h"
#include <map>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <dirent.h>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "acl/acl.h"

using namespace std;

namespace atlas_segmt_dlv3
{
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


    void Utils::ImageNhwc(shared_ptr<ImageDesc> &imageData, cv::Mat &nhwcImageChs, uint32_t size)
    {
        uint8_t *nchwBuf = new uint8_t[size];
        memcpy(static_cast<uint8_t *>(nchwBuf), nhwcImageChs.ptr<uint8_t>(0), size);

        imageData->size = size;
        imageData->data.reset((uint8_t *)nchwBuf, [](uint8_t *p) { delete[](p); });
    }


} // namespace atlas_segmt_dlv3