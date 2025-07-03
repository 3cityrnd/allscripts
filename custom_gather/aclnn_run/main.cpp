#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cassert>
#include <limits>
#include <cstdint>
#include <iostream>
#include <vector>
#include <fstream>
#include "acl/acl.h"

#include "acl/acl_op_compiler.h"
#include "aclnn_pimpek_gather.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

bool g_isDevice = false;
int deviceId = 0;

 void *workspace_;

void DestroyResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destroy resource failed");
    } else {
        INFO_LOG("Destroy resource success");
    }
}


aclTensor * CreateTensor(const std::vector<int64_t> shape, aclDataType dtype=ACL_FLOAT, void *data=nullptr) {
/*
typedef enum {
    ACL_DT_UNDEFINED = -1,  //未知数据类型，默认值
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT8 = 4,
    ACL_INT16 = 6,
    ACL_UINT16 = 7,
    ACL_UINT32 = 8,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_STRING = 13,
    ACL_COMPLEX64 = 16,
    ACL_COMPLEX128 = 17,
    ACL_BF16 = 27,
    ACL_INT4 = 29,
    ACL_UINT1 = 30,
    ACL_COMPLEX32 = 33,
} aclDataType;
*/

      int64_t size = 1;
        for(int64_t i=0;i<shape.size();i++)
        {
            size*=shape[i];
        }

        size*=sizeof(float);

       
      if(!data)
      {
        

       if (aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for [%zu] failed", size );
                return nullptr;
            }
            INFO_LOG("Create tensor size=[%zu] ", size);    
    
      }

        std::vector<int64_t> strides(shape.size(), 1);
          for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
       }  
       for(auto i : strides ) {
         INFO_LOG("Stride size=[%zu] ", i);    
       }

      aclTensor *inputTensor =
       aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                           ACL_FORMAT_ND, shape.data(), shape.size(), data);  


     if (inputTensor == nullptr) {
            ERROR_LOG("Create Tensor for input[%zu] failed", size);
            return nullptr;                      
     }

    return inputTensor;
}


int main(int argc, char **argv) {

    const char *json = "config.json";

    
    if(json)
    {
       std::ifstream file(json);
       if (!file) { 
        ERROR_LOG("Error file not found  %s\n", json);        
        return false; 
      }
    }

    if (aclInit(json) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);


    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        ERROR_LOG("Create stream failed");
        return false;
    }
    INFO_LOG("Create stream success");




     aclTensor* input_tensor = CreateTensor({3,3});  
     aclTensor* index_tensor = CreateTensor({2,2},ACL_INT32); 
     aclTensor* out = CreateTensor({2,2}); 
 
      
 // ./ascend-toolkit/latest/fwkacllib/include/aclnn/opdev/op_errno.h:#define ACLNN_ERR_PARAM_NULLPTR 161001

     if( !(index_tensor && index_tensor && out) )
     {
        ERROR_LOG("FATAL !! ERROR ");
     }

    size_t workspaceSize = 0;
    aclOpExecutor *handle = nullptr;
    auto ret =
        aclnnpimpek_gatherGetWorkspaceSize(input_tensor, index_tensor, -1, out,  &workspaceSize, &handle);
    if (ret != ACL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        ERROR_LOG("Get Operator Workspace failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }


    INFO_LOG("Execute aclnnpimpek_gatherGetWorkspaceSize success, workspace size %lu", workspaceSize);

    if (workspaceSize != 0) {
        if (aclrtMalloc(&workspace_, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory failed");
        }
    }


    ret = aclnnpimpek_gather(workspace_, workspaceSize, handle, stream);
    if (ret != ACL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        ERROR_LOG("Execute Operator failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Execute aclnnpimek_gather success");


    ret = aclrtSynchronizeStreamWithTimeout(stream, 5000);
    if (ret != SUCCESS) {
        ERROR_LOG("Synchronize stream failed. error code is %d", static_cast<int32_t>(ret));
        (void)aclrtDestroyStream(stream);
        return false;
    }
    INFO_LOG("Synchronize stream success");



   if(workspace_)
       aclrtFree(workspace_); 

#define FREETENSOR(x) \
     if(x) \
        aclDestroyTensor(x);

   FREETENSOR(input_tensor);
   FREETENSOR(index_tensor);
   FREETENSOR(out);



    if(stream)
    (void)aclrtDestroyStream(stream);


    DestroyResource();


    return SUCCESS;
}