#ifndef TENSOR_CU_H
#define TENSOR_CU_H
#include <mma.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace CudaTensorLib
{
    template <typename T> class componentTypes;

    template <> class componentTypes<unsigned char>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 1;};
        typedef unsigned char t;
        typedef unsigned char t1;
        typedef uchar2 t2;
    };

    template <> class componentTypes<uchar2>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 2;};
        typedef uchar2 t;
        typedef unsigned char t1;
        typedef uchar2 t2;
    };

    template <> class componentTypes<uchar4>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 4;};
        typedef uchar4 t;
        typedef unsigned char t1;
        typedef uchar2 t2;
    };

    template <> class componentTypes<half>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 1;};
        typedef half t;
        typedef half t1;
        typedef half2 t2;
    };

    template <> class componentTypes<half2>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 2;};
        typedef half2 t;
        typedef half t1;
        typedef half2 t2;
    };

    template <> class componentTypes<int>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 1;};
        typedef int t;
        typedef int t1;
        typedef int2 t2;
    };

    template <> class componentTypes<int2>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 2;};
        typedef int2 t;
        typedef int t1;
        typedef int2 t2;
    };

    template <> class componentTypes<int4>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 4;};
        typedef int4 t;
        typedef int t1;
        typedef int2 t2;
    };

    template <> class componentTypes<float>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 1;};
        typedef float t;
        typedef float t1;
        typedef float2 t2;
    };

    template <> class componentTypes<float2>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 2;};
        typedef float2 t;
        typedef float t1;
        typedef float2 t2;
    };

    template <> class componentTypes<float4>
    {
    public:
        static constexpr __device__ int getNumComponents(){return 4;};
        typedef float4 t;
        typedef float t1;
        typedef float2 t2;
    };

    template <typename mat_type, int M, int N, int K, typename T> class fragment
    {
    public:
        typedef componentTypes<T> t;
        static constexpr __device__ int getNumElementsPerComponentWithoutCopy() { return (std::is_same<mat_type, nvcuda::wmma::matrix_a>::value ? 1 : N) * (std::is_same<mat_type, nvcuda::wmma::matrix_b>::value ? 1 : M) * (std::is_same<mat_type, nvcuda::wmma::accumulator>::value ? 1 : K) / 32; };
        static constexpr __device__ int getNumElementsPerComponent() { return (std::is_same<typename t::t1, half>::value && (M * N == 256) && (!std::is_same<mat_type, nvcuda::wmma::accumulator>::value)) ? 16 : getNumElementsPerComponentWithoutCopy(); };
        static constexpr __device__ int getCopyPerThread() { return getNumElementsPerComponent() / getNumElementsPerComponentWithoutCopy(); };
        static constexpr __device__ int getNumComponents(){return componentTypes<T>::getNumComponents();};
        __device__ inline void fill(T data);
        typename t::t1 x[t::getNumComponents()][getNumElementsPerComponent()];
    };

    template<typename mat_type, int M, int N, int K, typename T>
    __device__ inline void fragment<mat_type, M, N, K, T>::fill(T data)
    {
#pragma unroll
        for (int element_id = 0; element_id < getNumComponents(); element_id++)
        {
#pragma unroll
            for (int component_id = 0; component_id < getNumElementsPerComponent(); component_id++)
            {
                x[element_id][component_id] = data;
            }
        }
    };

template <typename T, class DerivedClass>
class BaseManipulator
{
    public:
        __device__ inline BaseManipulator(){}
        using t = componentTypes<T>;

        template <typename regT, bool horizontalDirection, int regStride>
        __device__ inline void loadNxData2D(int x, int y, regT *reg)
        {
            typename t::t tmp_data = (horizontalDirection) ? ((DerivedClass *)this)->template loadData2D<typename t::t>(x, y) : ((DerivedClass *)this)->template loadData2D<typename t::t>(y, x);
            #pragma unroll
            for(unsigned int element_id = 0; element_id < t::getNumComponents(); element_id++)
            {
                ((typename componentTypes<regT>::t1 *)reg)[regStride * element_id] = (typename componentTypes<regT>::t1)(((typename t::t1 *)&tmp_data)[element_id]);
            }
        }
    
        template <typename regT, bool horizontalDirection, int regStride>
        __device__ inline void storeNxData2D(int x, int y, regT *reg)
        {
            typename t::t tmp_data;
            #pragma unroll
            for(unsigned int element_id = 0; element_id < t::getNumComponents(); element_id++)
            {
                ((typename t::t1 *)&tmp_data)[element_id] = (typename t::t1)(((typename componentTypes<regT>::t1 *)reg)[regStride * element_id]);
            }
    
            (horizontalDirection) ? ((DerivedClass *)this)->template storeData2D<typename t::t>(tmp_data, x, y) : ((DerivedClass *)this)->template storeData2D<typename t::t>(tmp_data, y, x);
        }
    
        template <typename regT, bool horizontalDirection>
        __device__ inline void loadNx4BData2D(int x, int y, regT *reg)
        {
            constexpr unsigned int vals_per_reg = 4/sizeof(typename componentTypes<regT>::t1);
            #pragma unroll
            for(unsigned int stride_x = 0; stride_x < vals_per_reg; stride_x++)
            {
                this->template loadNxData2D<regT, horizontalDirection, vals_per_reg>(x + stride_x, y, reg + stride_x);
            }
        }
    
        template <typename regT, bool horizontalDirection>
        __device__ inline void storeNx4BData2D(int x, int y, regT *reg)
        {
            constexpr unsigned int vals_per_reg = 4/sizeof(typename componentTypes<regT>::t1);
            #pragma unroll
            for(unsigned int stride_x = 0; stride_x < vals_per_reg; stride_x++)
            {
                this->template storeNxData2D<regT, horizontalDirection, vals_per_reg>(x + stride_x, y, reg + stride_x);
            }
        }
    
        template <typename regT, bool horizontalDirection, int elementsCount>
        __device__ inline void loadNxMData2D(int x, int y, regT *reg)
        {
            constexpr unsigned int vals_per_reg = elementsCount;
            #pragma unroll
            for(unsigned int stride_x = 0; stride_x < vals_per_reg; stride_x++)
            {
                this->template loadNxData2D<regT, horizontalDirection, vals_per_reg>(x + stride_x, y, reg + stride_x);
            }
        }
    
        template <typename regT, bool horizontalDirection, int elementsCount>
        __device__ inline void storeNxMData2D(int x, int y, regT *reg)
        {
            constexpr unsigned int vals_per_reg = elementsCount;
            #pragma unroll
            for(unsigned int stride_x = 0; stride_x < vals_per_reg; stride_x++)
            {
                this->template storeNxData2D<regT, horizontalDirection, vals_per_reg>(x + stride_x, y, reg + stride_x);
            }
        }
};

template <typename T, class ParentDerivedClassT = void>
class Linear2DManipulator : public BaseManipulator<T, typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Linear2DManipulator<T>, ParentDerivedClassT>::type>
{
public:
    using Linear2DDerivedClassT = typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Linear2DManipulator<T>, ParentDerivedClassT>::type;
    __device__ inline Linear2DManipulator(T *ptr, int x, int y, int stride): BaseManipulator<T,Linear2DDerivedClassT>(), ptr(ptr + x + y * stride), stride(stride){}
    int stride;
    T *ptr;

    template <typename outT>
    __device__ inline outT loadData2D(int x, int y = 0)
    {
        return (outT)(*((Linear2DDerivedClassT *)this)->getPtr2D(x, y));
    }

    template <typename inT>
    __device__ inline void storeData2D(inT data, int x, int y = 0)
    {
        (*((Linear2DDerivedClassT *)this)->getPtr2D(x, y)) = (typename BaseManipulator<T, Linear2DDerivedClassT>::t::t)data;
    }

    __device__ inline T *getPtr2D(int x, int y = 0)
    {
        return this->ptr + x + y * this->stride;
    }

    __device__ inline unsigned int load4BData2D(int x, int y = 0)
    {
        return ((unsigned int *)(((Linear2DDerivedClassT *)this)->getPtr2D(x,y)))[0];
    }


    __device__ inline void store4BData2D(unsigned int data, int x, int y = 0)
    {
        ((unsigned int *)(((Linear2DDerivedClassT *)this)->getPtr2D(x,y)))[0] = data;
    }

    __device__ inline typename BaseManipulator<T, Linear2DDerivedClassT>::t::t2 load2Data2D(int x, int y = 0)
    {
        return ((typename BaseManipulator<T, Linear2DDerivedClassT>::t::t2 *)(((Linear2DDerivedClassT *)this)->getPtr2D(x,y)))[0];
    }

    __device__ inline void store2Data2D(typename BaseManipulator<T, Linear2DDerivedClassT>::t::t2 data, int x, int y = 0)
    {
        ((typename BaseManipulator<T, Linear2DDerivedClassT>::t::t2 *)(((Linear2DDerivedClassT *)this)->getPtr2D(x,y)))[0] = data;
    }
};

enum class DimensionMapping3To2 { XY, XZ, YZ };

template <typename T, DimensionMapping3To2 dim_mapping, class ParentDerivedClassT = void>
class Linear3DManipulator : public Linear2DManipulator<T, typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Linear3DManipulator<T, dim_mapping>, ParentDerivedClassT>::type>
{
public:
    using Linear3DDerivedClassT = typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Linear3DManipulator<T, dim_mapping>, ParentDerivedClassT>::type;
    __device__ inline Linear3DManipulator(T *ptr, int x, int y, int z, int stride, int pitch): Linear2DManipulator<T, Linear3DDerivedClassT>(ptr + z * pitch, x, y, stride), pitch(pitch){}
    int pitch;

    template <typename outT>
    __device__ inline outT loadData3D(int x, int y = 0, int z = 0)
    {
        return (outT)(*this->getPtr3D(x, y, z));
    }

    template <typename inT>
    __device__ inline void storeData3D(inT data, int x = 0, int y = 0, int z = 0)
    {
        (*this->getPtr(x, y, z)) = (typename BaseManipulator<T, Linear3DDerivedClassT>::t::t)data;
    }

    __device__ inline unsigned int load4BData3D(int x = 0, int y = 0, int z = 0)
    {
        return ((unsigned int *)(this->getPtr3D(x,y,z)))[0];
    }

    __device__ inline void store4BData3D(unsigned int data, int x = 0, int y = 0, int z = 0)
    {
        ((unsigned int *)(this->getPtr3D(x,y,z)))[0] = data;
    }

    __device__ inline typename BaseManipulator<T, Linear3DDerivedClassT>::t::t2 load2Data3D(int x = 0, int y = 0, int z = 0)
    {
        return ((typename BaseManipulator<T, Linear3DDerivedClassT>::t::t2 *)(this->getPtr3D(x,y,z)))[0];
    }

    __device__ inline void store2Data3D(typename BaseManipulator<T, Linear3DDerivedClassT>::t::t2 data, int x = 0, int y = 0, int z = 0)
    {
        ((typename BaseManipulator<T, Linear3DDerivedClassT>::t::t2 *)(this->getPtr3D(x,y,z)))[0] = data;
    }

    __device__ inline int getXFrom2D(int id0, int id1)
    {
        return ((dim_mapping == DimensionMapping3To2::XY) || (dim_mapping == DimensionMapping3To2::XZ)) ? id0 : 0;
    }

    __device__ inline int getYFrom2D(int id0, int id1)
    {
        return (dim_mapping == DimensionMapping3To2::XY) ? id1 : ((dim_mapping == DimensionMapping3To2::YZ) ? id0 : 0);
    }

    __device__ inline int getZFrom2D(int id0, int id1)
    {
        return ((dim_mapping == DimensionMapping3To2::XZ) || (dim_mapping == DimensionMapping3To2::YZ)) ? id1 : 0;
    }

    __device__ inline T *getPtr2D(int id0 = 0, int id1 = 0)
    {
        return Linear2DManipulator<T, Linear3DDerivedClassT>(getXFrom2D(id0, id1), getYFrom2D(id0, id1)) + getZFrom2D(id0, id1) * this->pitch;
    }
};


template <typename T, typename ParentDerivedClassT = void>
class Block2DManipulator : public BaseManipulator<T, typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Block2DManipulator<T>, ParentDerivedClassT>::type>
{
public:
    using Block2DDerivedClassT = typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Block2DManipulator<T>, ParentDerivedClassT>::type;
    __device__ inline Block2DManipulator(int x, int y) : BaseManipulator<T, Block2DDerivedClassT>(), x(x), y(y){}
    int x;
    int y;
};

template <typename T, DimensionMapping3To2 dim_mapping, typename ParentDerivedClassT = void>
class Block3DManipulator : public Block2DManipulator<T, typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Block3DManipulator<T, dim_mapping>, ParentDerivedClassT>::type>
{
public:
    using Block3DDerivedClassT = typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Block3DManipulator<T, dim_mapping>, ParentDerivedClassT>::type;
    __device__ inline Block3DManipulator(int x, int y, int z) : Block2DManipulator<T, Block3DDerivedClassT>(x, y), z(z){}
    int z;

    __device__ inline int getXFrom2D(int id0, int id1)
    {
        return ((dim_mapping == DimensionMapping3To2::XY) || (dim_mapping == DimensionMapping3To2::XZ)) ? id0 : 0;
    }

    __device__ inline int getYFrom2D(int id0, int id1)
    {
        return (dim_mapping == DimensionMapping3To2::XY) ? id1 : ((dim_mapping == DimensionMapping3To2::YZ) ? id0 : 0);
    }

    __device__ inline int getZFrom2D(int id0, int id1)
    {
        return ((dim_mapping == DimensionMapping3To2::XZ) || (dim_mapping == DimensionMapping3To2::YZ)) ? id1 : 0;
    }

    template <typename outT>
    __device__ inline outT loadData2D(int id0 = 0, int id1 = 0)
    {
        return ((Block3DDerivedClassT *)this)->template loadData3D<outT>(this->getXFrom2D(id0, id1), this->getYFrom2D(id0, id1), this->getZFrom2D(id0, id1));
    }

    template <typename inT>
    __device__ inline void storeData2D(inT data, int id0 = 0, int id1 = 0)
    {
        return ((Block3DDerivedClassT *)this)->template storeData2D<inT>(data, this->getXFrom2D(id0, id1), this->getYFrom2D(id0, id1), this->getZFrom2D(id0, id1));
    }

    __device__ inline unsigned int load4BData2D(int id0 = 0, int id1 = 0)
    {
        return ((Block3DDerivedClassT *)this)->load4BData3D(this->getXFrom2D(id0, id1), this->getYFrom2D(id0, id1), this->getZFrom2D(id0, id1));
    }

    __device__ inline void store4BData2D(unsigned int data, int id0 = 0, int id1 = 0)
    {
        ((Block3DDerivedClassT *)this)->store4BData3D(data, this->getXFrom2D(id0, id1), this->getYFrom2D(id0, id1), this->getZFrom2D(id0, id1));
    }

    __device__ inline typename BaseManipulator<T, Block3DDerivedClassT>::t::t2 load2Data2D(int id0 = 0, int id1 = 0)
    {
        return ((Block3DDerivedClassT *)this)->load2Data3D(this->getXFrom2D(id0, id1), this->getYFrom2D(id0, id1), this->getZFrom2D(id0, id1));
    }

    __device__ inline void store2Data2D(typename BaseManipulator<T, Block3DDerivedClassT>::t::t2 data, int id0 = 0, int id1 = 0)
    {
        ((Block3DDerivedClassT *)this)->store2Data3D(data, this->getXFrom2D(id0, id1), this->getYFrom2D(id0, id1), this->getZFrom2D(id0, id1));
    }
};


template <typename T, DimensionMapping3To2 dim_mapping, typename ParentDerivedClassT = void>
class Surface3DWithOffsetsManipulator : public Block3DManipulator<T, dim_mapping, typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Surface3DWithOffsetsManipulator<T, dim_mapping>, ParentDerivedClassT>::type>
{
public:
    using Surface3DWithOffsetsDerivedClassT = typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Surface3DWithOffsetsManipulator<T, dim_mapping>, ParentDerivedClassT>::type;

    __device__ Surface3DWithOffsetsManipulator(cudaSurfaceObject_t *buffer, int2 *offsets, int x, int y, int z): Block3DManipulator<T, dim_mapping, Surface3DWithOffsetsDerivedClassT>(x, y, z), buffer(buffer), offsets(offsets){}

    __device__ inline unsigned int load4BData3D(int x, int y = 0, int z = 0)
    {
        constexpr unsigned int vals_per_reg = 4/sizeof(typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1);
        unsigned int data;
        #pragma unroll
        for(unsigned int stride_x = 0; stride_x < vals_per_reg; stride_x++)
        {
            ((typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1 *)&data)[stride_x] = ((Surface3DWithOffsetsDerivedClassT *)this)->template loadData3D<typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1>(x + stride_x, y, z);
        }
        return data;
    }

    __device__ inline void store4BData3D(unsigned int data, int x, int y = 0, int z = 0)
    {
        constexpr unsigned int vals_per_reg = 4/sizeof(typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1);
        #pragma unroll
        for(unsigned int stride_x = 0; stride_x < vals_per_reg; stride_x++)
        {
            ((Surface3DWithOffsetsDerivedClassT *)this)->template storeData3D<typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1>(((typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1 *)&data)[stride_x], x + stride_x, y, z);
        }
    }

    __device__ inline typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t2 load2Data3D(int x, int y = 0, int z = 0)
    {
        constexpr unsigned int vals_per_reg = 2;
        typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t2 data;
        #pragma unroll
        for(unsigned int stride_x = 0; stride_x < vals_per_reg; stride_x++)
        {
            ((typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1 *)&data)[stride_x] = ((Surface3DWithOffsetsDerivedClassT *)this)->template loadData3D<typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1>(x + stride_x, y, z);
        }
        return data;
    }

    __device__ inline void store2Data3D(typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t2 data, int x, int y = 0, int z = 0)
    {
        constexpr unsigned int vals_per_reg = 2;
        #pragma unroll
        for(unsigned int stride_x = 0; stride_x < vals_per_reg; stride_x++)
        {
            ((Surface3DWithOffsetsDerivedClassT *)this)->template storeData3D<typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1>(((typename BaseManipulator<T, Surface3DWithOffsetsDerivedClassT>::t::t1 *)&data)[stride_x], x + stride_x, y, z);
        }
    }

    cudaSurfaceObject_t *buffer;
    int2 *offsets;
};


template <typename T, DimensionMapping3To2 dim_mapping, typename ParentDerivedClassT = void>
class Layered3DSurfaceWithOffsetsManipulator : public Surface3DWithOffsetsManipulator<T, dim_mapping, typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Layered3DSurfaceWithOffsetsManipulator<T, dim_mapping>, ParentDerivedClassT>::type>
{
public:
    using Layered3DSurfaceWithOffsetsDerivedClassT = typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Layered3DSurfaceWithOffsetsManipulator<T, dim_mapping>, ParentDerivedClassT>::type;

    __device__ Layered3DSurfaceWithOffsetsManipulator(cudaSurfaceObject_t *buffer, int2 *offsets, int x, int y, int z): Surface3DWithOffsetsManipulator<T, dim_mapping, Layered3DSurfaceWithOffsetsDerivedClassT>(buffer, offsets, x, y, z){}
    template <typename outT>
    __device__ inline outT loadData3D(int x = 0, int y = 0, int z = 0)
    {
        return surf2DLayeredread<outT>(this->buffer, this->offsets[z].x + this->x + x, this->offsets[z].y + this->y + y, this->z + z, cudaBoundaryModeClamp);
    }

    template <typename inT>
    __device__ inline void storeData3D(inT data, int x = 0, int y = 0, int z = 0)
    {
        surf2DLayeredwrite<inT>(data, this->buffer, this->offsets[z].x + this->x + x, this->offsets[z].y + this->y + y, this->z + z);
    }
};

template <typename T, DimensionMapping3To2 dim_mapping, typename ParentDerivedClassT = void>
class Array3DSurfaceWithOffsetsManipulator : public Surface3DWithOffsetsManipulator<T, std::conditional<std::is_same<ParentDerivedClassT, void>::value, Array3DSurfaceWithOffsetsManipulator<T, dim_mapping>, ParentDerivedClassT>::type>
{
public:
    using Array3DSurfaceWithOffsetsDerivedClassT = typename std::conditional<std::is_same<ParentDerivedClassT, void>::value, Array3DSurfaceWithOffsetsManipulator<T, dim_mapping>, ParentDerivedClassT>::type;
    
    __device__ Array3DSurfaceWithOffsetsManipulator(cudaSurfaceObject_t *buffer, int2 *offsets, int x, int y, int z): Surface3DWithOffsetsManipulator<T, dim_mapping, Array3DSurfaceWithOffsetsDerivedClassT>(buffer, offsets, x, y, z){}
    template <typename outT>
    __device__ inline outT loadData3D(int x = 0, int y = 0, int z = 0)
    {
        return surf2Dread<outT>(this->buffer + this->z + z, this->offsets[z].x + this->x + x, this->offsets[z].y + this->y + y, cudaBoundaryModeClamp);
    }

    template <typename inT>
    __device__ inline void storeData3D(inT data, int x = 0, int y = 0, int z = 0)
    {
        surf2Dwrite<inT>(data, this->buffer + this->z + z, this->offsets[z].x + this->x + x, this->offsets[z].y + this->y + y);
    }
};


// load matrix a
template <typename mat_type, typename inT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_a>::value && componentTypes<matT>::getNumComponents() == 1, void>::type
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
            unsigned int id;
            if(MAT_B_Y_LOADS_PER_THREAD > 1) id = (y_stride_id & 1) + x_stride_id * 2 + (y_stride_id >> 1) * MAT_B_X_LOADS_PER_THREAD * 2;
            else id = x_stride_id;

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
                    unsigned int load_id = (std::is_same<MAJOR,nvcuda::wmma::row_major>::value) ? (x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER + act_y * stride) : ((x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER) * stride + act_y);
                    ((matT *)(&act_data))[x_offset] = (matT)(data[load_id]);
                }
            }

            #pragma unroll
            for(unsigned int copy_id = 0; copy_id < copy_per_thread; copy_id++)
            {
                ((unsigned int *)mat)[id + copy_id * copy_stride] = act_data;
            }
        }   
    }
}

template <typename mat_type, class DataManipulatorT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_a>::value, void>::type
load_matrix_sync(fragment<mat_type, M, N, K, matT> &mat, DataManipulatorT &data)
{
    constexpr unsigned int THREADS_PER_X = 4;
    constexpr unsigned int THREADS_PER_Y = 8;
    constexpr unsigned int MAT_B_X_VALUES_PER_REGISTER = 4/sizeof(typename componentTypes<matT>::t1);
    //int val_stride = stride/MAT_B_X_VALUES_PER_REGISTER;
    
    constexpr unsigned int MAT_B_X_LOADS_PER_THREAD = K/(THREADS_PER_X*MAT_B_X_VALUES_PER_REGISTER);
    constexpr unsigned int MAT_B_Y_LOADS_PER_THREAD = M/THREADS_PER_Y;
    unsigned int thread_in_warp = threadIdx.x%WARP_SIZE;
    unsigned int y_thread_start = thread_in_warp/THREADS_PER_X;
    unsigned int x_thread_start = thread_in_warp%THREADS_PER_X;
    constexpr unsigned int copy_stride = MAT_B_Y_LOADS_PER_THREAD*MAT_B_X_LOADS_PER_THREAD;
    constexpr unsigned int copy_per_thread = (std::is_same<half,matT>::value && (K == 16) && (M * N==256)) ? (8/copy_stride) : 1;
    constexpr bool is_load_linear = std::is_same<typename DataManipulatorT::t::t1,matT>::value && std::is_same<MAJOR,nvcuda::wmma::row_major>::value;

    // load from matrix b to fragments - indexing hell
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
            unsigned int act_data[mat.getNumComponents()];
            // fast path -> no conversion of datatypes or layouts for threads is needed
            if(is_load_linear)
            //if(false)
            {
                act_data[0] = data.load4BData2D(act_x * MAT_B_X_VALUES_PER_REGISTER, act_y);
            }
            else
            {
                data.template loadNx4BData2D<matT, std::is_same<MAJOR, nvcuda::wmma::row_major>::value>(act_x * MAT_B_X_VALUES_PER_REGISTER, act_y, (typename componentTypes<matT>::t1 *)act_data);
            }

            #pragma unroll
            for(unsigned int element_id = 0; element_id < mat.getNumComponents(); element_id++)
            {
                #pragma unroll
                for(unsigned int copy_id = 0; copy_id < copy_per_thread; copy_id++)
                {
                    ((unsigned int *)(mat.x[element_id]))[id + copy_id * copy_stride] = act_data[element_id];
                }
            }
        }   
    }
}

template <typename mat_type, typename inT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_a>::value && componentTypes<matT>::getNumComponents() == 1, void>::type
load_matrix_sync(CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, inT *data, unsigned int stride)
{
    load_matrix_sync<mat_type, inT, matT, M, N, K, MAJOR>(mat.x[0], data, stride);
}

/*template <typename mat_type, class DataManipulatorT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_a>::value, void>::type
load_matrix_sync(CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, DataManipulatorT &data_getter)
{
    load_matrix_sync<mat_type, DataManipulatorT, matT, M, N, K, MAJOR>(mat, data_getter);
}*/

// load matrix b
template <typename mat_type, typename inT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_b>::value && componentTypes<matT>::getNumComponents() == 1, void>::type
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
            unsigned int id;
            if(MAT_B_Y_LOADS_PER_THREAD > 1) id = x_stride_id + y_stride_id * MAT_B_X_LOADS_PER_THREAD;
            else id = x_stride_id;

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
                    unsigned int load_id = (std::is_same<MAJOR,nvcuda::wmma::col_major>::value) ? (x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER + act_y * stride) : ((x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER) * stride + act_y);
                    ((matT *)(&act_data))[x_offset] = (matT)(data[load_id]);
                }
            }

            #pragma unroll
            for(unsigned int copy_id = 0; copy_id < copy_per_thread; copy_id++)
            {
                ((unsigned int *)mat)[id + copy_id * copy_stride] = act_data;
            }
        }   
    }
}

template <typename mat_type, class DataManipulatorT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_b>::value, void>::type
load_matrix_sync(fragment<mat_type, M, N, K, matT>& mat, DataManipulatorT &data)
{
    constexpr unsigned int THREADS_PER_X = 4;
    constexpr unsigned int THREADS_PER_Y = 8;
    constexpr unsigned int MAT_B_X_VALUES_PER_REGISTER = 4/sizeof(matT);
    //int val_stride = stride/MAT_B_X_VALUES_PER_REGISTER;
    
    constexpr unsigned int MAT_B_X_LOADS_PER_THREAD = K/(THREADS_PER_X*MAT_B_X_VALUES_PER_REGISTER);
    constexpr unsigned int MAT_B_Y_LOADS_PER_THREAD = N/THREADS_PER_Y;
    unsigned int thread_in_warp = threadIdx.x%WARP_SIZE;
    unsigned int y_thread_start = thread_in_warp/THREADS_PER_X;
    unsigned int x_thread_start = thread_in_warp%THREADS_PER_X;
    constexpr unsigned int copy_stride = MAT_B_Y_LOADS_PER_THREAD*MAT_B_X_LOADS_PER_THREAD;
    constexpr unsigned int copy_per_thread = (std::is_same<half,matT>::value && (K == 16) && (M * N==256)) ? (8/copy_stride) : 1;
    constexpr bool is_load_linear = std::is_same<typename DataManipulatorT::t::t1,matT>::value && std::is_same<MAJOR,nvcuda::wmma::col_major>::value;

    // load from matrix a to fragments - indexing hell
    #pragma unroll
    for(unsigned int y_stride_id = 0; y_stride_id < MAT_B_Y_LOADS_PER_THREAD; y_stride_id++)
    {
        unsigned int act_y = y_thread_start + y_stride_id * THREADS_PER_Y;
        #pragma unroll
        for(unsigned int x_stride_id = 0; x_stride_id < MAT_B_X_LOADS_PER_THREAD; x_stride_id++)
        {
            unsigned int id;
            if(MAT_B_Y_LOADS_PER_THREAD > 1) id = x_stride_id + y_stride_id * MAT_B_X_LOADS_PER_THREAD;
            else id = x_stride_id;

            unsigned int act_x = x_thread_start + x_stride_id * THREADS_PER_X;
            unsigned int act_data[mat.getNumComponents()];
            if(is_load_linear)
            //if(false)
            {
                act_data[0] = data.load4BData2D(act_x * MAT_B_X_VALUES_PER_REGISTER, act_y);
            }
            else
            {
                data.template loadNx4BData2D<matT, std::is_same<MAJOR,nvcuda::wmma::col_major>::value>(act_x * MAT_B_X_VALUES_PER_REGISTER, act_y, (typename componentTypes<matT>::t1 *)act_data);
            }

            #pragma unroll
            for(unsigned int element_id = 0; element_id < mat.getNumComponents(); element_id++)
            {
                #pragma unroll
                for(unsigned int copy_id = 0; copy_id < copy_per_thread; copy_id++)
                {
                    ((unsigned int *)(mat.x[element_id]))[id + copy_id * copy_stride] = act_data[element_id];
                }
            }
        }   
    }
}

template <typename mat_type, typename inT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_b>::value && componentTypes<matT>::getNumComponents() == 1, void>::type
load_matrix_sync(CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, inT *data, unsigned int stride)
{
    load_matrix_sync<mat_type, inT, matT, M, N, K, MAJOR>(mat.x[0], data, stride);
}

/*template <typename mat_type, class DataManipulatorT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::matrix_b>::value, void>::type
load_matrix_sync(CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, DataManipulatorT &data_getter)
{
    load_matrix_sync<mat_type, DataManipulatorT, matT, M, N, K, MAJOR>(mat.x, data_getter);
}*/

// load matrix accumulator
template <typename mat_type, typename inT, typename matT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value && componentTypes<matT>::getNumComponents() == 1, void>::type
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
            unsigned int id;
            if(MAT_B_Y_LOADS_PER_THREAD > 1) id = (y_stride_id & 1) + x_stride_id * 2 + (y_stride_id >> 1) * MAT_B_X_LOADS_PER_THREAD * 2;
            else id = x_stride_id;

            unsigned int act_x = x_thread_start + x_stride_id * THREADS_PER_X;
            typename componentTypes<matT>::t2 act_data;
            // fast path -> no conversion of datatypes or layouts for threads is needed
            if(std::is_same<inT,matT>::value && (MAJOR == nvcuda::wmma::mem_row_major))
            {
                act_data = ((typename componentTypes<matT>::t2 *)data)[act_x + act_y * val_stride];
            }
            // slow path -> reindexing datatypes and/or conversion layouts for threads is needed
            else
            {
                #pragma unroll
                for(unsigned int x_offset = 0; x_offset < MAT_B_X_VALUES_PER_REGISTER; x_offset++)
                {
                    unsigned int load_id = (MAJOR == nvcuda::wmma::mem_row_major) ? (x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER + act_y * stride) : ((x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER) * stride + act_y);
                    ((matT *)(&act_data))[x_offset] = (matT)(data[load_id]);
                }
            }

            ((typename componentTypes<matT>::t2 *)mat)[id] = act_data;
        }   
    }
}

// load matrix accumulator
template <typename mat_type, class DataManipulatorT, typename matT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value, void>::type
load_matrix_sync(fragment<mat_type, M, N, K, matT>& mat, DataManipulatorT &data)
{
    constexpr unsigned int THREADS_PER_X = 4;
    constexpr unsigned int THREADS_PER_Y = 8;
    constexpr unsigned int MAT_B_X_VALUES_PER_REGISTER = 2;
    //int val_stride = stride/MAT_B_X_VALUES_PER_REGISTER;
    
    constexpr unsigned int MAT_B_X_LOADS_PER_THREAD = N/(THREADS_PER_X*MAT_B_X_VALUES_PER_REGISTER);
    constexpr unsigned int MAT_B_Y_LOADS_PER_THREAD = M/THREADS_PER_Y;
    unsigned int thread_in_warp = threadIdx.x%WARP_SIZE;
    unsigned int y_thread_start = thread_in_warp/THREADS_PER_X;
    unsigned int x_thread_start = thread_in_warp%THREADS_PER_X;
    constexpr bool is_load_linear = std::is_same<typename DataManipulatorT::t::t1,matT>::value && (MAJOR == nvcuda::wmma::mem_row_major);

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
            typename componentTypes<matT>::t2 act_data[mat.getNumComponents()];
            if(is_load_linear)
            {
                act_data[0] = data.load2Data2D(act_x * MAT_B_X_VALUES_PER_REGISTER, act_y);
            }
            else
            {
                data.template loadNxMData2D<matT, MAJOR == nvcuda::wmma::mem_row_major, 2>(act_x * MAT_B_X_VALUES_PER_REGISTER, act_y, (typename componentTypes<matT>::t1 *)act_data);
            }

            #pragma unroll
            for(unsigned int element_id = 0; element_id < mat.getNumComponents(); element_id++)
            {
                ((typename componentTypes<matT>::t2 *)(mat.x[element_id]))[id] = act_data[element_id];
            }
        }   
    }
}

template <typename mat_type, typename inT, typename matT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value && componentTypes<matT>::getNumComponents() == 1, void>::type
load_matrix_sync(CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, inT *data, unsigned int stride)
{
    load_matrix_sync<mat_type, inT, matT, M, N, K, MAJOR>(mat.x[0], data, stride);
}

/*template <typename mat_type, class DataManipulatorT, typename matT, int M, int N, int K, typename MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value, void>::type
load_matrix_sync(CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, DataManipulatorT &data_getter)
{
    load_matrix_sync<mat_type, DataManipulatorT, matT, M, N, K, MAJOR>(mat.x, data_getter);
}*/

// store matrix accumulator
template <typename mat_type, typename matT, typename outT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value && componentTypes<matT>::getNumComponents() == 1, void>::type
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
            typename componentTypes<matT>::t2 act_data = ((typename componentTypes<matT>::t2 *)mat)[id];
            // fast path -> no conversion of datatypes or layouts for threads is needed
            if(std::is_same<matT,outT>::value && (MAJOR == nvcuda::wmma::mem_row_major))
            {
                ((typename componentTypes<matT>::t2 *)data)[act_x + act_y * val_stride] = act_data;
            }
            // slow path -> reindexing datatypes and/or conversion layouts for threads is needed
            else
            {
                #pragma unroll
                for(unsigned int x_offset = 0; x_offset < MAT_B_X_VALUES_PER_REGISTER; x_offset++)
                {
                    unsigned int store_id = (MAJOR == nvcuda::wmma::mem_row_major) ? (x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER + act_y * stride) : ((x_offset + act_x * MAT_B_X_VALUES_PER_REGISTER) * stride + act_y);
                    data[store_id] = (outT)(((matT *)(&act_data))[x_offset]);
                }
            }
        }   
    }
}

template <typename mat_type, typename matT, class DataManipulatorT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value, void>::type
store_matrix_sync(DataManipulatorT &data, fragment<mat_type, M, N, K, matT>& mat)
{
    constexpr unsigned int THREADS_PER_X = 4;
    constexpr unsigned int THREADS_PER_Y = 8;
    constexpr unsigned int MAT_B_X_VALUES_PER_REGISTER = 2;
    
    constexpr unsigned int MAT_B_X_LOADS_PER_THREAD = N/(THREADS_PER_X*MAT_B_X_VALUES_PER_REGISTER);
    constexpr unsigned int MAT_B_Y_LOADS_PER_THREAD = M/THREADS_PER_Y;
    unsigned int thread_in_warp = threadIdx.x%WARP_SIZE;
    unsigned int y_thread_start = thread_in_warp/THREADS_PER_X;
    unsigned int x_thread_start = thread_in_warp%THREADS_PER_X;

    constexpr bool is_load_linear = std::is_same<typename DataManipulatorT::t::t1,matT>::value && (MAJOR == nvcuda::wmma::mem_row_major);
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

            typename componentTypes<matT>::t2 act_data[mat.getNumComponents()];
            #pragma unroll
            for (unsigned int element_id = 0; element_id < mat.getNumComponents(); element_id++)
            {
                act_data[element_id] = ((typename componentTypes<matT>::t2*)(mat.x[element_id]))[id];
            }
            // fast path -> no conversion of datatypes or layouts for threads is needed
            //if(is_load_linear)
            if(is_load_linear)
            {
                data.store2Data2D(act_data[0], act_x * MAT_B_X_VALUES_PER_REGISTER, act_y);
            }
            // slow path -> reindexing datatypes and/or conversion layouts for threads is needed
            else
            {
                data.template storeNxMData2D<matT, MAJOR == nvcuda::wmma::mem_row_major, MAT_B_X_VALUES_PER_REGISTER>(act_x * MAT_B_X_VALUES_PER_REGISTER, act_y, (typename componentTypes<matT>::t1 *)act_data);
            }
        }   
    }
}

template <typename mat_type, typename matT, typename outT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value && componentTypes<matT>::getNumComponents() == 1, void>::type
store_matrix_sync(outT *data, CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat, unsigned int stride)
{
    store_matrix_sync<mat_type, matT, outT, M, N, K, MAJOR>(data, mat.x[0], stride);
}

/*template <typename mat_type, typename matT, class DataManipulatorT, int M, int N, int K, nvcuda::wmma::layout_t MAJOR>
__device__ inline typename std::enable_if<std::is_same<mat_type, nvcuda::wmma::accumulator>::value, void>::type
store_matrix_sync(DataManipulatorT &data_getter, CudaTensorLib::fragment<mat_type, M, N, K, matT> &mat)
{
    store_matrix_sync<mat_type, matT, DataManipulatorT, M, N, K, MAJOR>(data_getter, mat.x);
}*/

template <typename T, int num_elements>
__device__ void fill_fragment(T *ptr, T data)
{
    #pragma unroll
    for(int i = 0; i < num_elements; i++)
    {
        ptr[i] = data;
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

#if /*!defined(__CUDA_ARCH__) ||*/ __CUDA_ARCH__ >= 800
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

template <typename matAccT, typename matT, int M, int N, int K>
__device__ typename std::enable_if<matT::t::getNumComponents() == matAccT::t::getNumComponents(), void>::type
mma_sync(fragment<nvcuda::wmma::accumulator, M, N, K, matAccT>& d,
         fragment<nvcuda::wmma::matrix_a, M, N, K, matT>& a,
         fragment<nvcuda::wmma::matrix_b, M, N, K, matT>& b,
         fragment<nvcuda::wmma::accumulator, M, N, K, matAccT>& c)
{
    #pragma unroll
    for (unsigned int component_id = 0; component_id < matT::t::getNumComponents(); component_id++)
    {
        mma_sync<matAccT, matT, M, N, K>(d.x[component_id], a.x[component_id], b.x[component_id], c.x[component_id]);
    }
}

}
#endif