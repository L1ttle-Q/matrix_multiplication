#ifndef __ERROR_CHECK_H__
#define __ERROR_CHECK_H__

#define CHECK(call){                                                    \
    cudaError_t e_code = call;                                          \
        if(e_code!=cudaSuccess){                                        \
            printf("##CUDA Error:\n");                                  \
            printf("  File: %s\n", __FILE__);                           \
            printf("  Line: %d\n", __LINE__);                           \
            printf("  Error info: %s\n", cudaGetErrorString(e_code));   \
        }                                                               \
}

#endif /* __ERROR_CHECK_H__ */