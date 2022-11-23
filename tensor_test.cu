#include "tensor.cu"
#define DEF_MULT_CONST 1

template <typename T, int num_elements>
__device__ void printMat(int id_in_warp, T *mat)
{
    for(int thread_id = 0; thread_id < WARP_SIZE; thread_id++)
    {
        if(id_in_warp == thread_id)
        {
            printf("%d: ", id_in_warp );
            for(int i = 0; i < num_elements; i++)
            {
                printf("%f ", (float)mat[i]);
            }
        printf("\n");
        }
    }
}

template <typename T, int X, int mat_size>
__device__ void printMatShare(T *data)
{
    for(int i = 0; i < mat_size; i++)
    {
        if (i % X == 0) printf("\n");
        printf("%f ", (float)(data[i]));
    }
    printf("\n\n");
}

template <typename matAccT, typename matT, int M, int N, int K, typename MAJOR_A, typename MAJOR_B, nvcuda::wmma::layout_t MAJOR_ACC, bool DEBUG_REGISTERS>
__device__ void checkMatsNative(int id_in_warp, matT *shared_a, matT *shared_b, matAccT *shared_acc)
{
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, matT, MAJOR_A> a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, matT, MAJOR_B> b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, matAccT> c;
    constexpr unsigned int stride_a = (std::is_same<MAJOR_A,nvcuda::wmma::row_major>::value) ? K : M;
    constexpr unsigned int stride_b = (std::is_same<MAJOR_B,nvcuda::wmma::row_major>::value) ? N : K;
    constexpr unsigned int stride_acc = (MAJOR_ACC == nvcuda::wmma::mem_row_major) ? N : M;

    nvcuda::wmma::load_matrix_sync(a, shared_a, stride_a);
    nvcuda::wmma::load_matrix_sync(b, shared_b, stride_b);
    nvcuda::wmma::fill_fragment(c, 0);
    nvcuda::wmma::mma_sync(c, a, b, c);
    nvcuda::wmma::store_matrix_sync(shared_acc, c, stride_acc, MAJOR_ACC);
    if(DEBUG_REGISTERS)
    {
        if (id_in_warp == 0) printf("Mat a\n");
        printMat<matT, nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, matT, MAJOR_A>::num_elements>(id_in_warp, a.x);
        if (id_in_warp == 0) printf("Mat b\n");
        printMat<matT, nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, matT, MAJOR_B>::num_elements>(id_in_warp, b.x);
        if (id_in_warp == 0) printf("Mat c\n");
        printMat<matAccT, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, matAccT>::num_elements>(id_in_warp, c.x);
    }
}



template <typename outT, typename inT, typename matAccT, typename matT, int M, int N, int K, typename MAJOR_A, typename MAJOR_B, nvcuda::wmma::layout_t MAJOR_ACC, bool DEBUG_REGISTERS>
__device__ void checkMatsCustom(int id_in_warp, inT *shared_a, inT *shared_b, outT *shared_acc)
{
    CudaTensorLib::fragment<nvcuda::wmma::matrix_a, M, N, K, matT> data_a;
    CudaTensorLib::fragment<nvcuda::wmma::matrix_b, M, N, K, matT> data_b;
    CudaTensorLib::fragment<nvcuda::wmma::accumulator, M, N, K, matAccT> data_acc;
    constexpr unsigned int stride_a = (std::is_same<MAJOR_A,nvcuda::wmma::row_major>::value) ? K : M;
    constexpr unsigned int stride_b = (std::is_same<MAJOR_B,nvcuda::wmma::row_major>::value) ? N : K;
    constexpr unsigned int stride_acc = (MAJOR_ACC == nvcuda::wmma::mem_row_major) ? N : M;
    CudaTensorLib::directLinear2DManipulator<inT> data_manipulator_a(shared_a, 0, 0, stride_a);
    CudaTensorLib::directLinear2DManipulator<inT> data_manipulator_b(shared_b, 0, 0, stride_b);
    CudaTensorLib::directLinear2DManipulator<outT> data_manipulator_acc(shared_acc, 0, 0, stride_acc);
    //if (id_in_warp == 0) printf("Ok %d\n", data_getter.stride);
    CudaTensorLib::load_matrix_sync<nvcuda::wmma::matrix_a, CudaTensorLib::directLinear2DManipulator<inT>, matT, M, N, K, MAJOR_A>(data_a, data_manipulator_a);
    CudaTensorLib::load_matrix_sync<nvcuda::wmma::matrix_b, CudaTensorLib::directLinear2DManipulator<inT>, matT, M, N, K, MAJOR_B>(data_b, data_manipulator_b);
    //CudaTensorLib::load_matrix_sync<nvcuda::wmma::matrix_a, inT, matT, M, N, K, MAJOR_A>(data_a, shared_a, stride_a);
    //CudaTensorLib::load_matrix_sync<nvcuda::wmma::matrix_b, inT, matT, M, N, K, MAJOR_B>(data_b, shared_b, stride_b);
    CudaTensorLib::fill_fragment<matAccT, CudaTensorLib::fragment<nvcuda::wmma::accumulator, M, N, K, matAccT>>(data_acc, (matAccT)0);
    CudaTensorLib::mma_sync<matAccT, matT, M, N, K>(data_acc.x, data_a.x, data_b.x, data_acc.x);
    //CudaTensorLib::store_matrix_sync<nvcuda::wmma::accumulator, matAccT, outT, M, N, K, MAJOR_ACC>(shared_acc, data_acc, stride_acc);
    CudaTensorLib::store_matrix_sync<nvcuda::wmma::accumulator, matAccT, CudaTensorLib::directLinear2DManipulator<outT>, M, N, K, MAJOR_ACC>(data_manipulator_acc, data_acc);
    if(DEBUG_REGISTERS)
    {
        if (id_in_warp == 0) printf("Mat a\n");
        printMat<matT, data_a.num_elements>(id_in_warp, data_a.x);
        if (id_in_warp == 0) printf("Mat b\n");
        printMat<matT, data_b.num_elements>(id_in_warp, data_b.x);
        if (id_in_warp == 0) printf("Mat c\n");
        printMat<matAccT, data_acc.num_elements>(id_in_warp, data_acc.x);
    }
}

template <typename T1, typename T2, int W, int H>
__device__ bool compare_mat(T1 *data1, T2 *data2)
{
    bool result = true;
    for(int i = 0; i < W*H; i++)
    {
        result &= ((float)data1[i] == (float)data2[i]);
        if((float)data1[i] != (float)data2[i]) printf("%d:%f!=%f\n", i, (float)data1[i], (float)data2[i]);
    }
    if(!result)
    {
        printMatShare<T1,W,W*H>(data1);
        printMatShare<T2,W,W*H>(data2);
    }
    return result;
}

template <typename outT1, typename inT1, typename outT2, typename inT2, int M, int N, int K>
__device__ bool compare_mats(unsigned char *data1, unsigned char *data2)
{
    bool result = true;
    printf("\n\nMatrix A:\n");
    result = compare_mat<inT1,inT2,K,M>((inT1 *)data1, (inT2 *)data2) && result;
    printf("\n\nMatrix B:\n");
    result = compare_mat<inT1,inT2,K,N>(((inT1 *)data1) + M*K,((inT2 *)data2) + M*K) && result;
    printf("\n\nMatrix C:\n");
    result = compare_mat<outT1,outT2,N,M>((outT1 *)(((inT1 *)data1) + M*K + N*K),(outT2 *)(((inT2 *)data2) + M*K + N*K)) && result;
    return result;
}

template <typename T, int data_size>
__device__ void fill_mem_by_range(T *data, int id_in_warp)
{
    for(int i = id_in_warp; i < data_size; i += WARP_SIZE)
    {
        data[i] = (T)i;
    }
}

__device__ void test_mat_shared(int2 coords, void *localMemory)
{
    constexpr int M = 8;
    constexpr int N = 32;
    constexpr int K = 16;
    typedef unsigned char matT;
    typedef int matAccT;
    typedef unsigned char inT1;
    typedef int outT1; 
    typedef unsigned char inT2;
    typedef int outT2; 
 
    matT *mat_native_a = (matT *)localMemory;
    matT *mat_native_b = mat_native_a + M*K;
    matAccT *mat_native_acc = (matAccT *)(mat_native_b + N*K);
    inT1 *mat_custom1_a = (inT1 *)(mat_native_acc + N*M);
    inT1 *mat_custom1_b = mat_custom1_a + M*K;
    outT1 *mat_custom1_acc = (outT1 *)(mat_custom1_b + N*K);
    inT2 *mat_custom2_a = (inT2 *)(mat_custom1_acc + N*M);
    inT2 *mat_custom2_b = mat_custom2_a + M*K;
    outT2 *mat_custom2_acc = (outT2 *)(mat_custom2_b + N*K);

    fill_mem_by_range<matT, K*M>(mat_native_a, coords.x);
    fill_mem_by_range<matT, N*K>(mat_native_b, coords.x);
    fill_mem_by_range<inT1, K*M>(mat_custom1_a, coords.x);
    fill_mem_by_range<inT1, N*K>(mat_custom1_b, coords.x);
    fill_mem_by_range<inT2, K*M>(mat_custom2_a, coords.x);
    fill_mem_by_range<inT2, N*K>(mat_custom2_b, coords.x);

    if((coords.x >= WARP_SIZE) || (coords.y > 0)) return;
    checkMatsNative<matAccT, matT, M, N, K, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, false>(coords.x, mat_native_a, mat_native_b, mat_native_acc);
    checkMatsCustom<outT1, inT1, matAccT, matT, M, N, K, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, false>(coords.x, mat_custom1_a, mat_custom1_b, mat_custom1_acc);
    checkMatsCustom<outT2, inT2, matAccT, matT, M, N, K, nvcuda::wmma::col_major, nvcuda::wmma::row_major, nvcuda::wmma::mem_row_major, false>(coords.x, mat_custom2_a, mat_custom2_b, mat_custom2_acc);
    __syncthreads();
    if((coords.x == 0) && (coords.y == 0))
    {
        bool native_custom_result = compare_mat<matAccT, outT1, N, M>(mat_native_acc,mat_custom1_acc);
        bool custom_result = compare_mat<outT1, outT2, N, M>(mat_custom1_acc,mat_custom2_acc);
        if(!native_custom_result)
            printf("\n\nNative and custom implementation has differend reuslts.\n\n");
        if(!custom_result)
            printf("\n\nCustom implementations has differend reuslts.\n\n");
    }
}

template <typename inT, typename outT, typename matT, typename matAccT, int M, int N, int K, typename MAJOR_A, typename MAJOR_B, nvcuda::wmma::layout_t MAJOR_ACC>
__device__ void test_mat_global_warp(outT *c, inT *a, inT *b, int m, int n, int k)
{
    int warp_in_group = threadIdx.x/32;
    int2 warp_id;
    warp_id.x = blockIdx.x * WARPS_PER_ROW + (warp_in_group%WARPS_PER_ROW);
    warp_id.y = blockIdx.y * WARPS_PER_COL + (warp_in_group/WARPS_PER_ROW);
    CudaTensorLib::fragment<nvcuda::wmma::matrix_a, M, N, K, matT> data_a;
    CudaTensorLib::fragment<nvcuda::wmma::matrix_b, M, N, K, matT> data_b;
    CudaTensorLib::fragment<nvcuda::wmma::accumulator, M, N, K, matAccT> data_acc;
    constexpr unsigned int stride_a = (std::is_same<MAJOR_A,nvcuda::wmma::row_major>::value) ? k : m;
    constexpr unsigned int stride_b = (std::is_same<MAJOR_B,nvcuda::wmma::row_major>::value) ? n : k;
    constexpr unsigned int stride_acc = (MAJOR_ACC == nvcuda::wmma::mem_row_major) ? n : m;
    inT *a_ptr = a + ((std::is_same<MAJOR_A,nvcuda::wmma::row_major>::value) ? k : 1) * warp_id.y * M;
    inT *b_ptr = b + ((std::is_same<MAJOR_B,nvcuda::wmma::row_major>::value) ? 1 : k) * warp_id.x * N;
    outT *c_ptr = c + ((MAJOR_ACC == nvcuda::wmma::mem_row_major) ? 1 : m) * warp_id.x * N + ((MAJOR_ACC == nvcuda::wmma::mem_row_major) ? n : 1) * warp_id.y * M;
    unsigned int a_shift = ((std::is_same<MAJOR_A,nvcuda::wmma::row_major>::value) ? k : 1) * K;
    unsigned int b_shift = ((std::is_same<MAJOR_B,nvcuda::wmma::row_major>::value) ? 1 : k) * K;

    CudaTensorLib::fill_fragment<matAccT, CudaTensorLib::fragment<nvcuda::wmma::accumulator, M, N, K, matAccT>>(data_acc, (matAccT)0);

    for(int i = 0; i < k/K; i++)
    {
        CudaTensorLib::load_matrix_sync<nvcuda::wmma::matrix_a, inT, matT, M, N, K, MAJOR_A>(data_a, a_ptr + i * a_shift, stride_a);
        CudaTensorLib::load_matrix_sync<nvcuda::wmma::matrix_b, inT, matT, M, N, K, MAJOR_B>(data_b, b_ptr + i * b_shift, stride_b);
        CudaTensorLib::mma_sync<matAccT, matT, M, N, K>(data_acc.x, data_a.x, data_b.x, data_acc.x);
    }
 
    CudaTensorLib::store_matrix_sync<nvcuda::wmma::accumulator, matAccT, outT, M, N, K, MAJOR_ACC>(c_ptr, data_c, stride_acc);
}