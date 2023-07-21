#pragma once

#include <iomanip>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include "time_measure_util.h"

// struct __device_builtin__ __align__(2*sizeof(unsigned long int)) encoded_lattice_pt {
template<int N>
struct encoded_lattice_pt {
    std::array<uint64_t, N> data;

    static encoded_lattice_pt<N> create_empty_stencil() {
        encoded_lattice_pt sten = encoded_lattice_pt<N>();
        for (int i = 0; i != N; ++i)
        {
            sten.data[i] = ~0;
        }
        return sten;
    }

    __host__ __device__ encoded_lattice_pt()
    {
        for (int i = 0; i != N; ++i)
            data[i] = 0;
    }
    
    __host__ __device__ bool operator==(const encoded_lattice_pt& other) const
    {
        for (int i = 0; i != N; ++i)
        {
            if (data[i] != other.data[i])
                return false;
        }
        return true;
    }

    // __host__ __device__ bool operator<(const encoded_lattice_pt& other) const
    // {
    //     for (int i = N; i > 0; i--)
    //     {
    //         if (data[i] != other.data[i])
    //             return data[i] < other.data[i];
    //     }
    //     return data[0] < other.data[0];
    // }

    __host__ __device__ inline const uint64_t& get_portion(int& start_bit, int& end_bit) const
    {
        assert(end_bit > start_bit);
        const int index = start_bit / 64;
        assert((end_bit - 1) / 64 == index);
        start_bit = start_bit % 64;
        end_bit = end_bit % 64;
        return data[index];
    }

    __host__ __device__ inline uint64_t& get_portion(int& start_bit, int& end_bit)
    {
        assert(end_bit > start_bit);
        const int index = start_bit / 64;
        assert((end_bit - 1) / 64 == index);
        start_bit = start_bit % 64;
        end_bit = end_bit % 64;
        return data[index];
    }

    __host__ __device__ inline int extract_number(int start_bit, int end_bit) const
    {
        const uint64_t portion = get_portion(start_bit, end_bit);
        return ((portion >> start_bit) & ((1 << end_bit - start_bit) - 1));
    }

    __host__ __device__ inline void put_number(int start_bit, int end_bit, uint64_t number)
    {
        uint64_t& portion = get_portion(start_bit, end_bit);
        const uint64_t remove_mask = ~(((1 << end_bit - start_bit) - 1) << start_bit);
        portion = (portion & remove_mask) | (number << start_bit);
    }

    __host__ __device__ inline bool increment(int start_bit, int end_bit)
    {
        uint64_t& portion = get_portion(start_bit, end_bit);
        portion += ((uint64_t) 1) << start_bit;
        if (((portion >> start_bit) & ((1 << end_bit - start_bit) - 1)) == 0) // overflow
            return true;
        return false;
    }

    __host__ __device__ inline bool decrement(int start_bit, int end_bit)
    {
        uint64_t& portion = get_portion(start_bit, end_bit);
        if (((portion >> start_bit) & ((1 << end_bit - start_bit) - 1)) == 0) // underflow
            return true;
        portion -= ((uint64_t) 1) << start_bit;
        return false;
    }
};

template<int N>
std::ostream& operator<<(std::ostream& o, const encoded_lattice_pt<N>& pt)
{
    for (int i = 0; i != N; ++i)
        o << pt.data[i] <<", ";
    return o;
}

inline int get_cuda_device()
{   
    return 0; // Get first possible GPU. CUDA_VISIBLE_DEVICES automatically masks the rest of GPUs.
}

inline void print_gpu_memory_stats()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout<<"Total memory(MB): "<<total / (1024 * 1024)<<", Free(MB): "<<free / (1024 * 1024)<<std::endl;
}

inline void checkCudaError(cudaError_t status, std::string errorMsg)
{
    if (status != cudaSuccess) {
        std::cout << "CUDA error: " << errorMsg << ", status" <<cudaGetErrorString(status) << std::endl;
        throw std::exception();
    }
}

inline void initialize_gpu(bool verbose = false)
{
    const int cuda_device = get_cuda_device();
    cudaSetDevice(cuda_device);
    if (verbose)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cuda_device);
        std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << ", device number " << cuda_device << "\n";
    }
}

template <typename T, typename ITERATOR>
inline thrust::device_vector<T> duplicate_by_counts(const ITERATOR values_begin, const thrust::device_vector<uint32_t>& counts)
{
    MEASURE_FUNCTION_EXECUTION_TIME;
    thrust::device_vector<uint32_t> counts_sum(counts.size() + 1);
    counts_sum[0] = 0;
    thrust::inclusive_scan(counts.begin(), counts.end(), counts_sum.begin() + 1);
    
    int out_size = counts_sum.back();
    thrust::device_vector<uint32_t> output_indices(out_size, 0);

    thrust::scatter(
        thrust::make_constant_iterator<uint32_t>(1), thrust::make_constant_iterator<uint32_t>(1) + counts.size(), 
        counts_sum.begin(), output_indices.begin());

    thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin());
    thrust::transform(output_indices.begin(), output_indices.end(), thrust::make_constant_iterator(1), output_indices.begin(), thrust::minus<uint32_t>());

    thrust::device_vector<T> out_values(out_size);
    thrust::gather(output_indices.begin(), output_indices.end(), values_begin, out_values.begin());

    return out_values;
}

template<typename T>
inline void print_vector(const thrust::device_vector<T>& v, const char* name, const int num = 0)
{
    std::cout<<name<<": ";
    if (num == 0)
        thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    else
    {
        int size = std::distance(v.begin(), v.end());
        thrust::copy(v.begin(), v.begin() + std::min(size, num), std::ostream_iterator<T>(std::cout, " "));
    }
    std::cout<<"\n";
}

template<typename T>
inline void print_vector(const thrust::device_ptr<T>& v, const char* name, const int num)
{
    std::cout<<name<<": ";
    thrust::copy(v, v + num, std::ostream_iterator<T>(std::cout, " "));
    std::cout<<"\n";
}

template<typename T>
inline void print_matrix(const thrust::device_vector<T>& v, const char* name, const int num_cols)
{
    std::cout<<name<<":\n";
    const int num_rows = v.size() / num_cols;
    auto start_location = v.begin();
    for (int r = 0; r != num_rows; r++)
    {
        std::vector<T> row(start_location, start_location + num_cols);
        for (auto val : row)
            std::cout << std::setw(2) << val << " ";
        // thrust::copy(start_location, start_location + num_cols, std::ostream_iterator<T>(std::cout, " "));

        start_location += num_cols;
        std::cout<<"\n";
    }
}

template<typename T>
inline void print_matrix(const thrust::device_ptr<T>& v, const char* name, const int num_cols, const int num_rows)
{
    std::cout<<name<<":\n";
    auto start_location = v;
    for (int r = 0; r != num_rows; r++)
    {
        std::vector<T> row(start_location, start_location + num_cols);
        for (auto val : row)
            std::cout << std::setw(2) << val << " ";

        // thrust::copy(start_location, start_location + num_cols, std::ostream_iterator<T>(std::cout, " "));
        start_location += num_cols;
        std::cout<<"\n";
    }
}

__host__ __device__ inline int positive_modulo(int i, int n) {
    return (i % n + n) % n;
}

__host__ __device__ inline int floor_divisor(int i, int n) {
    return i / n - (i % n < 0 ? 1 : 0); // E.g. produces -1 for i = -10, and n = 17
}

// reminder: vertex of canonical simplex with given reminder.
// dim_index: which coordinate to compute. 
// Since simplex lives in (d_pos + 1) dimensions so both reminder, dim_index should be < than d_pos + 1.
__host__ __device__ inline int compute_canonical_simplex_point_coordinate(const int reminder, const int dim_index, const int d_pos)
{
    assert(reminder >= 0);
    assert(reminder < d_pos + 1);
    assert(dim_index >= 0);
    assert(dim_index < d_pos + 1);
    return reminder - ((dim_index + reminder - d_pos > 0) ? d_pos + 1 : 0);
}

template<typename T>
__host__ __device__ inline T encode_point(
    const int* const cumulative_num_bits, const int* const min_c, const int* const rem0, const int* const ranks, 
    const int batch_index, const int start_index, const int reminder, const int d_lattice)
    {
        T packedNumber;
        int start_bit = 0;
        int end_bit = cumulative_num_bits[0];
        for (int c = 0; c != d_lattice + 1; c++)
        {
            if (c < d_lattice - 1)
            {
                const int current_rank = ranks[start_index + c];
                const int current_rem0 = rem0[start_index + c];

                int pt_coordinate = current_rem0 + compute_canonical_simplex_point_coordinate(reminder, current_rank, d_lattice - 1);
                if (pt_coordinate != 0)
                    pt_coordinate = floor_divisor(pt_coordinate, d_lattice);
                pt_coordinate -= min_c[c];
                assert(pt_coordinate >= 0);
                packedNumber.put_number(start_bit, end_bit, pt_coordinate);
            }
            else if (c == d_lattice - 1)
                packedNumber.put_number(start_bit, end_bit, reminder);
            else
                packedNumber.put_number(start_bit, end_bit, batch_index);
            start_bit = end_bit;
            end_bit = cumulative_num_bits[c + 1];
        }
        return packedNumber;
    }

template<typename T>
__host__ __device__ inline void decode_point(const int* const cumulative_num_bits, const int* const min_c,
    const int out_index_pt, const int d_lattice, const T encoded_point, int* output)
    {
        const int batch_index = encoded_point.extract_number(cumulative_num_bits[d_lattice - 1], cumulative_num_bits[d_lattice]);
        const int reminder = encoded_point.extract_number(cumulative_num_bits[d_lattice - 2], cumulative_num_bits[d_lattice - 1]);
        int output_index = out_index_pt * d_lattice;
        output[output_index++] = batch_index;

        int start_bit = 0;
        int end_bit = cumulative_num_bits[0];
        for (int index_d = 0; index_d != d_lattice - 1; index_d++)
        {
            int current_number = encoded_point.extract_number(start_bit, end_bit) + min_c[index_d];
            current_number *= d_lattice;

            output[output_index++] = current_number + reminder;
            start_bit = end_bit;
            end_bit = cumulative_num_bits[index_d + 1];
        }
    }

template<typename T>
__host__ __device__ inline void compute_neighbour_encoding(const int* const cumulative_num_bits,
    const int d_lattice, const T encoded_point, const int direction, T& neighbour_point_plus, T& neighbour_point_minus,
    bool& plus_overflow, bool& minus_overflow)
    {
        plus_overflow = false;
        minus_overflow = false;
        // for dit in range(n_ch_1):
        // offset = [n_ch if i == dit else -1 for i in range(n_ch)]
        const int start_bit_batch_index = cumulative_num_bits[d_lattice - 1];
        const int start_bit_reminder = cumulative_num_bits[d_lattice - 2];

        const int cur_reminder = encoded_point.extract_number(start_bit_reminder, start_bit_batch_index);
        
        const int nplus_reminder = cur_reminder == 0 ? d_lattice - 1: cur_reminder - 1;
        const int nminus_reminder = cur_reminder == d_lattice - 1 ? 0: cur_reminder + 1;

        neighbour_point_plus.put_number(start_bit_reminder, start_bit_batch_index, nplus_reminder);
        neighbour_point_minus.put_number(start_bit_reminder, start_bit_batch_index, nminus_reminder);
        
        if (cur_reminder == 0)
        {
            int start_bit = 0;
            int end_bit = cumulative_num_bits[0];
            for (int c = 0; c != d_lattice - 1; c++)
            {
                if (c == direction) // due to -n_ch
                    minus_overflow |= neighbour_point_minus.decrement(start_bit, end_bit);
                else // due to -1
                    plus_overflow |= neighbour_point_plus.decrement(start_bit, end_bit);
                start_bit = end_bit;
                end_bit = cumulative_num_bits[c + 1];
            }
        }
        else if (cur_reminder == d_lattice - 1)
        {
            int start_bit = 0;
            int end_bit = cumulative_num_bits[0];
            for (int c = 0; c != d_lattice - 1; c++)
            {
                if (c == direction)
                    plus_overflow |= neighbour_point_plus.increment(start_bit, end_bit);
                else
                    minus_overflow |= neighbour_point_minus.increment(start_bit, end_bit);
                start_bit = end_bit;
                end_bit = cumulative_num_bits[c + 1];
            }
        }
        else if(direction != d_lattice - 1)
        {
            const int start_bit = direction > 0 ? cumulative_num_bits[direction - 1] : 0;
            const int end_bit = cumulative_num_bits[direction];   
            plus_overflow |= neighbour_point_plus.increment(start_bit, end_bit);
            minus_overflow |= neighbour_point_minus.decrement(start_bit, end_bit);
        }
    }
