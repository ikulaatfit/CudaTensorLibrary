namespace CudaTensorLib
{

    template <typename mat_type, int M, int N, int K, typename T> class fragment;

    template <int M, int N, int K, typename T> class fragment<nvcuda::wmma::matrix_a, M, N, K, T>
    {
    public:
        enum {num_elements = M*K/32};
        T x[M*K/32];
    };
    
    template <int M, int N, int K, typename T> class fragment<nvcuda::wmma::matrix_b, M, N, K, T>
    {
    public:
        enum {num_elements = N*K/32};
        T x[N*K/32];
    };
    
    template <int M, int N, int K, typename T> class fragment<nvcuda::wmma::accumulator, M, N, K, T>
    {
    public:
        enum {num_elements = N*M/32};
        T x[N*M/32];
    };
    
    template <> class fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half>
    {
    public:
        enum {num_elements = 16};
        half x[16];
    };
    template <> class fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half>
    {
    public:
        enum {num_elements = 16};
        half x[16];
    };
    template <> class fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half>
    {
    public:
        enum {num_elements = 16};
        half x[16];
    };
    template <> class fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half>
    {
    public:
        enum {num_elements = 16};
        half x[16];
    };
    template <> class fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half>
    {
    public:
        enum {num_elements = 16};
        half x[16];
    };
    template <> class fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half>
    {
    public:
        enum {num_elements = 16};
        half x[16];
    };

    template <typename T> class componentType2;

    template <> class componentType2<half>
    {
    public:
    typedef half2 t;
    };

    template <> class componentType2<int>
    {
    public:
    typedef int2 t;
    };

    template <> class componentType2<float>
    {
    public:
    typedef float2 t;
    };


// load matrix a
template <typename mat_type, typename inT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_a>::value, void>::type
load_matrix_sync(matT *mat, inT *data, unsigned int stride)
{
    constexpr unsigned int THREADS_PER_X = 4;
    constexpr unsigned int THREADS_PER_Y = 8;
    constexpr unsigned int MAT_B_X_VALUES_PER_REGISTER = 4/sizeof(matT);
    int val_stride = stride/MAT_B_X_VALUES_PER_REGISTER;
    
    constexpr unsigned int MAT_B_X_LOADS_PER_THREAD = K/(THREADS_PER_X*MAT_B_X_VALUES_PER_REGISTER);
    constexpr unsigned int MAT_B_Y_LOADS_PER_THREAD = M/THREADS_PER_Y;
    unsigned int thread_in_warp = threadIdx.x%WARP_SIZE;
    unsigned int y_thread_start = thread_in_warp/THREADS_PER_X;
    unsigned int x_thread_start = thread_in_warp%THREADS_PER_X;
    constexpr unsigned int copy_stride = MAT_B_Y_LOADS_PER_THREAD*MAT_B_X_LOADS_PER_THREAD;
    constexpr unsigned int copy_per_thread = (std::is_same<half,matT>::value && (K == 16) && (M * N==256)) ? (8/copy_stride) : 1;

    // load from matrix b to fragments - indexing hell
    #pragma unroll
    for(unsigned int y_stride_id = 0; y_stride_id < MAT_B_Y_LOADS_PER_THREAD; y_stride_id++)
    {
        unsigned int act_y = y_thread_start + y_stride_id * THREADS_PER_Y;
        #pragma unroll
        for(unsigned int x_stride_id = 0; x_stride_id < MAT_B_X_LOADS_PER_THREAD; x_stride_id++)
        {
            unsigned int act_x = x_thread_start + x_stride_id * THREADS_PER_X;
            unsigned int act_data;
            // fast path -> no conversion of datatypes or layouts for threads is needed
            if(std::is_same<inT,matT>::value && std::is_same<MAJOR,nvcuda::wmma::row_major>::value)
            {
                act_data = ((unsigned int *)data)[act_x + act_y * val_stride];
            }
            else
            {
                #pragma unroll
                for(unsigned int x_offset = 0; x_offset < MAT_B_X_VALUES_PER_REGISTER; x_offset++)
                {
                    unsigned int load_id = (std::is_same<MAJOR,nvcuda::wmma::row_major>::value) ? (x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER + act_y * val_stride * MAT_B_X_VALUES_PER_REGISTER) : ((x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER) * val_stride * MAT_B_X_VALUES_PER_REGISTER + act_y);
                    ((matT *)(&act_data))[x_offset] = (matT)(data[load_id]);
                }
            }
            unsigned int id;
            if(MAT_B_Y_LOADS_PER_THREAD > 1) id = (y_stride_id & 1) + x_stride_id * 2 + (y_stride_id >> 1) * MAT_B_X_LOADS_PER_THREAD * 2;
            else id = x_stride_id;

            #pragma unroll
            for(unsigned int copy_id = 0; copy_id < copy_per_thread; copy_id++)
            {
                ((unsigned int *)mat)[id + copy_id * copy_stride] = act_data;
            }
        }   
    }
}

template <typename mat_type, typename inT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_a>::value, void>::type
load_matrix_sync(CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, inT *data, unsigned int stride)
{
    load_matrix_sync<mat_type, inT, matT, M, N, K, MAJOR>(mat.x, data, stride);
}
// load matrix b
template <typename mat_type, typename inT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_b>::value, void>::type
load_matrix_sync(matT *mat, inT *data, unsigned int stride)
{
    constexpr unsigned int THREADS_PER_X = 4;
    constexpr unsigned int THREADS_PER_Y = 8;
    constexpr unsigned int MAT_B_X_VALUES_PER_REGISTER = 4/sizeof(matT);
    int val_stride = stride/MAT_B_X_VALUES_PER_REGISTER;
    
    constexpr unsigned int MAT_B_X_LOADS_PER_THREAD = K/(THREADS_PER_X*MAT_B_X_VALUES_PER_REGISTER);
    constexpr unsigned int MAT_B_Y_LOADS_PER_THREAD = N/THREADS_PER_Y;
    unsigned int thread_in_warp = threadIdx.x%WARP_SIZE;
    unsigned int y_thread_start = thread_in_warp/THREADS_PER_X;
    unsigned int x_thread_start = thread_in_warp%THREADS_PER_X;
    constexpr unsigned int copy_stride = MAT_B_Y_LOADS_PER_THREAD*MAT_B_X_LOADS_PER_THREAD;
    constexpr unsigned int copy_per_thread = (std::is_same<half,matT>::value && (K == 16) && (M * N==256)) ? (8/copy_stride) : 1;

    // load from matrix a to fragments - indexing hell
    #pragma unroll
    for(unsigned int y_stride_id = 0; y_stride_id < MAT_B_Y_LOADS_PER_THREAD; y_stride_id++)
    {
        unsigned int act_y = y_thread_start + y_stride_id * THREADS_PER_Y;
        #pragma unroll
        for(unsigned int x_stride_id = 0; x_stride_id < MAT_B_X_LOADS_PER_THREAD; x_stride_id++)
        {
            unsigned int act_x = x_thread_start + x_stride_id * THREADS_PER_X;
            unsigned int act_data;
            if(std::is_same<inT,matT>::value && std::is_same<MAJOR,nvcuda::wmma::col_major>::value)
            {
                act_data = ((unsigned int *)data)[act_x + act_y * val_stride];
            }
            else
            {
                #pragma unroll
                for(unsigned int x_offset = 0; x_offset < MAT_B_X_VALUES_PER_REGISTER; x_offset++)
                {
                    unsigned int load_id = (std::is_same<MAJOR,nvcuda::wmma::col_major>::value) ? (x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER + act_y * val_stride * MAT_B_X_VALUES_PER_REGISTER) : ((x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER) * val_stride * MAT_B_X_VALUES_PER_REGISTER + act_y);
                    ((matT *)(&act_data))[x_offset] = (matT)(data[load_id]);
                }
            }
            unsigned int id;
            if(MAT_B_Y_LOADS_PER_THREAD > 1) id = x_stride_id + y_stride_id * MAT_B_X_LOADS_PER_THREAD;
            else id = x_stride_id;

            #pragma unroll
            for(unsigned int copy_id = 0; copy_id < copy_per_thread; copy_id++)
            {
                ((unsigned int *)mat)[id + copy_id * copy_stride] = act_data;
            }
        }   
    }
}

template <typename mat_type, typename inT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_b>::value, void>::type
load_matrix_sync(CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, inT *data, unsigned int stride)
{
    load_matrix_sync<mat_type, inT, matT, M, N, K, MAJOR>(mat.x, data, stride);
}

// load matrix accumulator
template <typename mat_type, typename inT, typename matT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value, void>::type
load_matrix_sync(matT *mat, inT *data, unsigned int stride)
{
    constexpr unsigned int THREADS_PER_X = 4;
    constexpr unsigned int THREADS_PER_Y = 8;
    constexpr unsigned int MAT_B_X_VALUES_PER_REGISTER = 2;
    int val_stride = stride/MAT_B_X_VALUES_PER_REGISTER;
    
    constexpr unsigned int MAT_B_X_LOADS_PER_THREAD = N/(THREADS_PER_X*MAT_B_X_VALUES_PER_REGISTER);
    constexpr unsigned int MAT_B_Y_LOADS_PER_THREAD = M/THREADS_PER_Y;
    unsigned int thread_in_warp = threadIdx.x%WARP_SIZE;
    unsigned int y_thread_start = thread_in_warp/THREADS_PER_X;
    unsigned int x_thread_start = thread_in_warp%THREADS_PER_X;

    #pragma unroll
    for(unsigned int y_stride_id = 0; y_stride_id < MAT_B_Y_LOADS_PER_THREAD; y_stride_id++)
    {
        unsigned int act_y = y_thread_start + y_stride_id * THREADS_PER_Y;
        #pragma unroll
        for(unsigned int x_stride_id = 0; x_stride_id < MAT_B_X_LOADS_PER_THREAD; x_stride_id++)
        {
            unsigned int act_x = x_thread_start + x_stride_id * THREADS_PER_X;
            componentType2<matT>::t act_data;
            // fast path -> no conversion of datatypes or layouts for threads is needed
            if(std::is_same<inT,matT>::value && (MAJOR == nvcuda::wmma::mem_row_major))
            {
                act_data = ((componentType2<matT>::t *)data)[act_x + act_y * val_stride];
            }
            // slow path -> reindexing datatypes and/or conversion layouts for threads is needed
            else
            {
                #pragma unroll
                for(unsigned int x_offset = 0; x_offset < MAT_B_X_VALUES_PER_REGISTER; x_offset++)
                {
                    unsigned int load_id = (MAJOR == nvcuda::wmma::mem_row_major) ? (x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER + act_y * val_stride * MAT_B_X_VALUES_PER_REGISTER) : ((x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER) * val_stride * MAT_B_X_VALUES_PER_REGISTER + act_y);
                    ((matT *)(&act_data))[x_offset] = (matT)(data[load_id]);
                }
            }
            unsigned int id;
            if(MAT_B_Y_LOADS_PER_THREAD > 1) id = (y_stride_id & 1) + x_stride_id * 2 + (y_stride_id >> 1) * MAT_B_X_LOADS_PER_THREAD * 2;
            else id = x_stride_id;

            ((componentType2<matT>::t *)mat)[id] = act_data;
        }   
    }
}

template <typename mat_type, typename inT, typename matT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value, void>::type
load_matrix_sync(CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, inT *data, unsigned int stride)
{
    load_matrix_sync<mat_type, inT, matT, M, N, K, MAJOR>(mat.x, data, stride);
}

// store matrix accumulator
template <typename mat_type, typename matT, typename outT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value, void>::type
store_matrix_sync(outT *data, matT *mat, unsigned int stride)
{
    constexpr unsigned int THREADS_PER_X = 4;
    constexpr unsigned int THREADS_PER_Y = 8;
    constexpr unsigned int MAT_B_X_VALUES_PER_REGISTER = 2;
    int val_stride = stride/MAT_B_X_VALUES_PER_REGISTER;
    
    constexpr unsigned int MAT_B_X_LOADS_PER_THREAD = N/(THREADS_PER_X*MAT_B_X_VALUES_PER_REGISTER);
    constexpr unsigned int MAT_B_Y_LOADS_PER_THREAD = M/THREADS_PER_Y;
    unsigned int thread_in_warp = threadIdx.x%WARP_SIZE;
    unsigned int y_thread_start = thread_in_warp/THREADS_PER_X;
    unsigned int x_thread_start = thread_in_warp%THREADS_PER_X;

    // load from accumulator matrix to fragments - indexing hell
    #pragma unroll
    for(unsigned int y_stride_id = 0; y_stride_id < MAT_B_Y_LOADS_PER_THREAD; y_stride_id++)
    {
        unsigned int act_y = y_thread_start + y_stride_id * THREADS_PER_Y;
        #pragma unroll
        for(unsigned int x_stride_id = 0; x_stride_id < MAT_B_X_LOADS_PER_THREAD; x_stride_id++)
        {
            unsigned int id;
            if(MAT_B_Y_LOADS_PER_THREAD > 1) id = (y_stride_id & 1) + x_stride_id * 2 + (y_stride_id >> 1) * MAT_B_X_LOADS_PER_THREAD * 2;
            else id = x_stride_id;
            unsigned int act_x = x_thread_start + x_stride_id * THREADS_PER_X;
            componentType2<matT>::t act_data = ((componentType2<matT>::t *)mat)[id];
            // fast path -> no conversion of datatypes or layouts for threads is needed
            if(std::is_same<matT,outT>::value && (MAJOR == nvcuda::wmma::mem_row_major))
            {
                ((componentType2<matT>::t *)data)[act_x + act_y * val_stride] = act_data;
            }
            // slow path -> reindexing datatypes and/or conversion layouts for threads is needed
            else
            {
                #pragma unroll
                for(unsigned int x_offset = 0; x_offset < MAT_B_X_VALUES_PER_REGISTER; x_offset++)
                {
                    unsigned int store_id = (MAJOR == nvcuda::wmma::mem_row_major) ? (x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER + act_y * val_stride * MAT_B_X_VALUES_PER_REGISTER) : ((x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER) * val_stride * MAT_B_X_VALUES_PER_REGISTER + act_y);
                    data[store_id] = (outT)(((matT *)(&act_data))[x_offset]);
                }
            }
        }   
    }
}

template <typename mat_type, typename matT, typename outT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value, void>::type
store_matrix_sync(outT *data, CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, unsigned int stride)
{
    store_matrix_sync<mat_type, matT, outT, M, N, K, MAJOR>(data, mat.x, stride);
}

template <typename T, int num_elements>
__device__ void fill_fragment(T *ptr, T data)
{
    #pragma unroll
    for(int i = 0; i < num_elements; i++)
    {
        ptr[i] = data;
    }
}

template <typename T, typename matT>
__device__ void fill_fragment(matT &ptr, T data)
{
    #pragma unroll
    for(int i = 0; i < ptr.num_elements; i++)
    {
        ptr.x[i] = data;
    }
}

template <typename matAccT, typename matT, int M, int N, int K> __device__ void mma_sync(matAccT *d, matT *a, matT *b, matAccT *c);

template <> __device__ void mma_sync<half, half, 16, 16, 16>(half *d, half *a, half *b, half *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    unsigned int *c_ptr = (unsigned int *)c;
    unsigned int *d_ptr = (unsigned int *)d;
    asm("wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19}, {%20,%21,%22,%23};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(a_ptr[4]), "r"(a_ptr[5]), "r"(a_ptr[6]), "r"(a_ptr[7]),
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]), "r"(b_ptr[4]), "r"(b_ptr[5]), "r"(b_ptr[6]), "r"(b_ptr[7]),
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]));
}

template <> __device__ void mma_sync<float, half, 16, 16, 16>(float *d, half *a, half *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = c;
    float *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27,%28,%29,%30,%31};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]),"=f"(d_ptr[4]),"=f"(d_ptr[5]),"=f"(d_ptr[6]),"=f"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(a_ptr[4]), "r"(a_ptr[5]), "r"(a_ptr[6]), "r"(a_ptr[7]),
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]), "r"(b_ptr[4]), "r"(b_ptr[5]), "r"(b_ptr[6]), "r"(b_ptr[7]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]), "f"(c_ptr[4]), "f"(c_ptr[5]), "f"(c_ptr[6]), "f"(c_ptr[7]));
}

template <> __device__ void mma_sync<half, half, 32, 8, 16>(half *d, half *a, half *b, half *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    unsigned int *c_ptr = (unsigned int *)c;
    unsigned int *d_ptr = (unsigned int *)d;
    asm("wmma.mma.sync.aligned.row.col.m32n8k16.f16.f16 {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19}, {%20,%21,%22,%23};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(a_ptr[4]), "r"(a_ptr[5]), "r"(a_ptr[6]), "r"(a_ptr[7]),
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]), "r"(b_ptr[4]), "r"(b_ptr[5]), "r"(b_ptr[6]), "r"(b_ptr[7]),
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]));
}

template <> __device__ void mma_sync<float, half, 32, 8, 16>(float *d, half *a, half *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = c;
    float *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m32n8k16.f32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27,%28,%29,%30,%31};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]),"=f"(d_ptr[4]),"=f"(d_ptr[5]),"=f"(d_ptr[6]),"=f"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(a_ptr[4]), "r"(a_ptr[5]), "r"(a_ptr[6]), "r"(a_ptr[7]),
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]), "r"(b_ptr[4]), "r"(b_ptr[5]), "r"(b_ptr[6]), "r"(b_ptr[7]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]), "f"(c_ptr[4]), "f"(c_ptr[5]), "f"(c_ptr[6]), "f"(c_ptr[7]));
}

template <> __device__ void mma_sync<half, half, 8, 32, 16>(half *d, half *a, half *b, half *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    unsigned int *c_ptr = (unsigned int *)c;
    unsigned int *d_ptr = (unsigned int *)d;
    asm("wmma.mma.sync.aligned.row.col.m8n32k16.f16.f16 {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19}, {%20,%21,%22,%23};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(a_ptr[4]), "r"(a_ptr[5]), "r"(a_ptr[6]), "r"(a_ptr[7]),
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]), "r"(b_ptr[4]), "r"(b_ptr[5]), "r"(b_ptr[6]), "r"(b_ptr[7]),
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]));
}

template <> __device__ void mma_sync<float, half, 8, 32, 16>(float *d, half *a, half *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = c;
    float *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m8n32k16.f32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27,%28,%29,%30,%31};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]),"=f"(d_ptr[4]),"=f"(d_ptr[5]),"=f"(d_ptr[6]),"=f"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(a_ptr[4]), "r"(a_ptr[5]), "r"(a_ptr[6]), "r"(a_ptr[7]),
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]), "r"(b_ptr[4]), "r"(b_ptr[5]), "r"(b_ptr[6]), "r"(b_ptr[7]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]), "f"(c_ptr[4]), "f"(c_ptr[5]), "f"(c_ptr[6]), "f"(c_ptr[7]));
}

template <> __device__ void mma_sync<half, half, 16, 8, 8>(half *d, half *a, half *b, half *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    unsigned int *c_ptr = (unsigned int *)c;
    unsigned int *d_ptr = (unsigned int *)d;
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]),
         "r"(b_ptr[0]), "r"(c_ptr[0]),
         "r"(c_ptr[1]));
}

template <> __device__ void mma_sync<float, half, 16, 8, 8>(float *d, half *a, half *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = (float *)c;
    float *d_ptr = (float *)d;
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]),
         "r"(b_ptr[0]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]));
}





#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
template <> __device__ void mma_sync<int, unsigned char, 16, 16, 16>(int *d, unsigned char *a, unsigned char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m16n16k16.s32.u8.u8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]),"=r"(d_ptr[4]),"=r"(d_ptr[5]),"=r"(d_ptr[6]),"=r"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]),
         "r"(b_ptr[0]), "r"(b_ptr[1]),
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]), "r"(c_ptr[4]), "r"(c_ptr[5]), "r"(c_ptr[6]), "r"(c_ptr[7]));
}

template <> __device__ void mma_sync<int, char, 16, 16, 16>(int *d, char *a, char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]),"=r"(d_ptr[4]),"=r"(d_ptr[5]),"=r"(d_ptr[6]),"=r"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]),
         "r"(b_ptr[0]), "r"(b_ptr[1]),
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]), "r"(c_ptr[4]), "r"(c_ptr[5]), "r"(c_ptr[6]), "r"(c_ptr[7]));
}

template <> __device__ void mma_sync<int, unsigned char, 32, 8, 16>(int *d, unsigned char *a, unsigned char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m32n8k16.s32.u8.u8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12}, {%13,%14,%15,%16,%17,%18,%19,%20};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]),"=r"(d_ptr[4]),"=r"(d_ptr[5]),"=r"(d_ptr[6]),"=r"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]),
         "r"(b_ptr[0]), 
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]), "r"(c_ptr[4]), "r"(c_ptr[5]), "r"(c_ptr[6]), "r"(c_ptr[7]));
}

template <> __device__ void mma_sync<int, char, 32, 8, 16>(int *d, char *a, char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m32n8k16.s32.s8.s8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12}, {%13,%14,%15,%16,%17,%18,%19,%20};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]),"=r"(d_ptr[4]),"=r"(d_ptr[5]),"=r"(d_ptr[6]),"=r"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]),
         "r"(b_ptr[0]), 
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]), "r"(c_ptr[4]), "r"(c_ptr[5]), "r"(c_ptr[6]), "r"(c_ptr[7]));
}

template <> __device__ void mma_sync<int, unsigned char, 8, 32, 16>(int *d, unsigned char *a, unsigned char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m8n32k16.s32.u8.u8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8}, {%9,%10,%11,%12}, {%13,%14,%15,%16,%17,%18,%19,%20};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]),"=r"(d_ptr[4]),"=r"(d_ptr[5]),"=r"(d_ptr[6]),"=r"(d_ptr[7]) :
         "r"(a_ptr[0]), 
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]),
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]), "r"(c_ptr[4]), "r"(c_ptr[5]), "r"(c_ptr[6]), "r"(c_ptr[7]));
}

template <> __device__ void mma_sync<int, char, 8, 32, 16>(int *d, char *a, char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m8n32k16.s32.s8.s8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8}, {%9,%10,%11,%12}, {%13,%14,%15,%16,%17,%18,%19,%20};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]),"=r"(d_ptr[4]),"=r"(d_ptr[5]),"=r"(d_ptr[6]),"=r"(d_ptr[7]) :
         "r"(a_ptr[0]), 
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]),
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]), "r"(c_ptr[4]), "r"(c_ptr[5]), "r"(c_ptr[6]), "r"(c_ptr[7]));
}

template <> __device__ void mma_sync<int, unsigned char, 8, 8, 16>(int *d, unsigned char *a, unsigned char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]) :
         "r"(a_ptr[0]),
         "r"(b_ptr[0]),
         "r"(c_ptr[0]), "r"(c_ptr[1]));
}

template <> __device__ void mma_sync<int, char, 8, 8, 16>(int *d, char *a, char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]) :
         "r"(a_ptr[0]),
         "r"(b_ptr[0]),
         "r"(c_ptr[0]), "r"(c_ptr[1]));
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
template <> __device__ void mma_sync<half, half, 16, 8, 16>(half *d, half *a, half *b, half *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    unsigned int *c_ptr = (unsigned int *)c;
    unsigned int *d_ptr = (unsigned int *)d;
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]),
         "r"(b_ptr[0]), "r"(b_ptr[1]),
         "r"(c_ptr[0]), "r"(c_ptr[1]));
}

template <> __device__ void mma_sync<float, __nv_bfloat16, 16, 8, 16>(float *d, __nv_bfloat16 *a, __nv_bfloat16 *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = (float *)c;
    float *d_ptr = (float *)d;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]),
         "r"(b_ptr[0]), "r"(b_ptr[1]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]));
}

template <> __device__ void mma_sync<float, half, 16, 8, 16>(float *d, half *a, half *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = (float *)c;
    float *d_ptr = (float *)d;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]),
         "r"(b_ptr[0]), "r"(b_ptr[1]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]));
}

template <> __device__ void mma_sync<int, unsigned char, 16, 8, 16>(int *d, unsigned char *a, unsigned char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]),
         "r"(b_ptr[0]),
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]));
}

template <> __device__ void mma_sync<int, char, 16, 8, 16>(int *d, char *a, char *b, int *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    int *c_ptr = c;
    int *d_ptr = d;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};" :
        "=r"(d_ptr[0]),"=r"(d_ptr[1]),"=r"(d_ptr[2]),"=r"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]),
         "r"(b_ptr[0]),
         "r"(c_ptr[0]), "r"(c_ptr[1]), "r"(c_ptr[2]), "r"(c_ptr[3]));
}

template <> __device__ void mma_sync<float, __nv_bfloat16, 16, 8, 8>(float *d, __nv_bfloat16 *a, __nv_bfloat16 *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = (float *)c;
    float *d_ptr = (float *)d;
    asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]),
         "r"(b_ptr[0]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]));
}

template <> __device__ void mma_sync<float, precision::tf32, 16, 8, 8>(float *d, precision::tf32 *a, precision::tf32 *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = (float *)c;
    float *d_ptr = (float *)d;
    asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]),
         "r"(b_ptr[0]), "r"(b_ptr[1]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]));
}

template <> __device__ void mma_sync<float, __nv_bfloat16, 16, 16, 16>(float *d, __nv_bfloat16 *a, __nv_bfloat16 *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = c;
    float *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27,%28,%29,%30,%31};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]),"=f"(d_ptr[4]),"=f"(d_ptr[5]),"=f"(d_ptr[6]),"=f"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(a_ptr[4]), "r"(a_ptr[5]), "r"(a_ptr[6]), "r"(a_ptr[7]),
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]), "r"(b_ptr[4]), "r"(b_ptr[5]), "r"(b_ptr[6]), "r"(b_ptr[7]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]), "f"(c_ptr[4]), "f"(c_ptr[5]), "f"(c_ptr[6]), "f"(c_ptr[7]));
}

template <> __device__ void mma_sync<float, __nv_bfloat16, 32, 8, 16>(float *d, __nv_bfloat16 *a, __nv_bfloat16 *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = c;
    float *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m32n8k16.f32.bf16.bf16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27,%28,%29,%30,%31};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]),"=f"(d_ptr[4]),"=f"(d_ptr[5]),"=f"(d_ptr[6]),"=f"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(a_ptr[4]), "r"(a_ptr[5]), "r"(a_ptr[6]), "r"(a_ptr[7]),
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]), "r"(b_ptr[4]), "r"(b_ptr[5]), "r"(b_ptr[6]), "r"(b_ptr[7]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]), "f"(c_ptr[4]), "f"(c_ptr[5]), "f"(c_ptr[6]), "f"(c_ptr[7]));
}

template <> __device__ void mma_sync<float, __nv_bfloat16, 8, 32, 16>(float *d, __nv_bfloat16 *a, __nv_bfloat16 *b, float *c)
{
    unsigned int *a_ptr = (unsigned int *)a;
    unsigned int *b_ptr = (unsigned int *)b;
    float *c_ptr = c;
    float *d_ptr = d;
    asm("wmma.mma.sync.aligned.row.col.m8n32k16.f32.bf16.bf16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27,%28,%29,%30,%31};" :
        "=f"(d_ptr[0]),"=f"(d_ptr[1]),"=f"(d_ptr[2]),"=f"(d_ptr[3]),"=f"(d_ptr[4]),"=f"(d_ptr[5]),"=f"(d_ptr[6]),"=f"(d_ptr[7]) :
         "r"(a_ptr[0]), "r"(a_ptr[1]), "r"(a_ptr[2]), "r"(a_ptr[3]), "r"(a_ptr[4]), "r"(a_ptr[5]), "r"(a_ptr[6]), "r"(a_ptr[7]),
         "r"(b_ptr[0]), "r"(b_ptr[1]), "r"(b_ptr[2]), "r"(b_ptr[3]), "r"(b_ptr[4]), "r"(b_ptr[5]), "r"(b_ptr[6]), "r"(b_ptr[7]),
         "f"(c_ptr[0]), "f"(c_ptr[1]), "f"(c_ptr[2]), "f"(c_ptr[3]), "f"(c_ptr[4]), "f"(c_ptr[5]), "f"(c_ptr[6]), "f"(c_ptr[7]));
}
#endif

}