#pragma once

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/adjacent_difference.h>

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
        thrust::copy(start_location, start_location + num_cols, std::ostream_iterator<T>(std::cout, " "));
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
        thrust::copy(start_location, start_location + num_cols, std::ostream_iterator<T>(std::cout, " "));
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
        T packedNumber = 0;
        int c = 0;
        int shift = 0;
        for (c = 0; c != d_lattice - 1; c++)
        {
            const int current_rank = ranks[start_index + c];
            const int current_rem0 = rem0[start_index + c];

            int pt_coordinate = current_rem0 + compute_canonical_simplex_point_coordinate(reminder, current_rank, d_lattice - 1);
            if (pt_coordinate != 0)
                pt_coordinate = floor_divisor(pt_coordinate, d_lattice);
            pt_coordinate -= min_c[c];
            assert(pt_coordinate >= 0);
            // Pack the number by shifting it and combining with the packedNumber
            packedNumber |= ((T) pt_coordinate) << shift;
            shift = cumulative_num_bits[c];
        }
        packedNumber |= ((T) reminder) << shift;
        packedNumber |= ((T) batch_index) << cumulative_num_bits[d_lattice - 1];
        return packedNumber;
    }

template<typename T>
__host__ __device__ inline void decode_point(const int* const cumulative_num_bits, const int* const min_c,
    const int out_index_pt, const int d_lattice, const T encoded_point, int* output)
    {
        T masked = encoded_point;
        const int batch_index = masked >> cumulative_num_bits[d_lattice - 1]; 
        masked = masked & ~(~0 << cumulative_num_bits[d_lattice - 1]);
        int output_index = out_index_pt * d_lattice;
        output[output_index++] = batch_index;

        const int reminder = masked >> cumulative_num_bits[d_lattice - 2]; 
        int shift = 0;
        for (int index_d = 0; index_d != d_lattice - 1; index_d++)
        {
            int next_shift = cumulative_num_bits[index_d];
            int current_number = ((masked >> shift) & ((1 << next_shift - shift) - 1)) + min_c[index_d];
            current_number *= d_lattice;

            output[output_index++] = current_number + reminder;
            shift = next_shift;
        }
    }

template<typename T>
__host__ __device__ inline void compute_neighbour_encoding(const int* const cumulative_num_bits,
    const int d_lattice, const T encoded_point, const int direction, T& neighbour_point_plus, T& neighbour_point_minus,
    bool& plus_overflow, bool& minus_overflow)
    {
        // for dit in range(n_ch_1):
        // offset = [n_ch if i == dit else -1 for i in range(n_ch)]
        const int num_bits_reminder = cumulative_num_bits[d_lattice - 2];
        const T cur_reminder = (encoded_point & ((1 << cumulative_num_bits[d_lattice - 1]) - 1)) >> num_bits_reminder; 
        const T nplus_reminder = cur_reminder == 0 ? d_lattice - 1: cur_reminder - 1;
        const T nminus_reminder = cur_reminder == d_lattice - 1 ? 0: cur_reminder + 1;

        const T reminder_remove_mask = ~(cur_reminder << num_bits_reminder);
        neighbour_point_plus = (encoded_point & reminder_remove_mask) | (nplus_reminder << num_bits_reminder);
        neighbour_point_minus = (encoded_point & reminder_remove_mask) | (nminus_reminder << num_bits_reminder);
        plus_overflow = false;
        minus_overflow = false;
        if (cur_reminder == 0)
        {
            int shift = 0;
            for (int c = 0; c != d_lattice - 1; c++)
            {
                const int next_shift = cumulative_num_bits[c];
                const int current_number = ((encoded_point >> shift) & ((1 << next_shift - shift) - 1));
                const T op = 1 << shift;
                if (c == direction)
                {
                    minus_overflow |= current_number == 0;
                    neighbour_point_minus -= op;
                }
                else
                {
                    plus_overflow |= current_number == 0;
                    neighbour_point_plus -= op;
                }
                shift = next_shift;
            }
        }
        else if (cur_reminder == d_lattice - 1)
        {
            // [-1 3 -1] -> [0, 1, 0]
            // [0 0 0] -> [1, 1, 1]
            int shift = 0;
            for (int c = 0; c != d_lattice - 1; c++)
            {
                const int next_shift = cumulative_num_bits[c];
                const int current_number = ((encoded_point >> shift) & ((1 << next_shift - shift) - 1));
                const T op = 1 << shift;
                if (c == direction)
                {
                    minus_overflow |= current_number == 0;
                    neighbour_point_plus -= op;
                }
                else
                {
                    minus_overflow |= current_number == next_shift - shift;
                    neighbour_point_minus += op;
                }
                shift = next_shift;
            }
        }
        else if(direction != d_lattice - 1)
        {
            const int shift = direction > 0 ? cumulative_num_bits[direction - 1] : 0;
            const int next_shift = cumulative_num_bits[direction];   
            const int current_number = ((encoded_point >> shift) & ((1 << next_shift - shift) - 1));
            const T op = 1 << shift;
            plus_overflow |= current_number == next_shift - shift;
            neighbour_point_plus += op;
            minus_overflow |= current_number == 0;
            neighbour_point_minus -= op;
        }
        // printf("cur_reminder: %d, nplus_reminder: %d, nminus_reminder: %d\n", 
        //     cur_reminder, 
        //     (neighbour_point_plus & ((1 << cumulative_num_bits[d_lattice - 1]) - 1)) >> num_bits_reminder, 
        //     (neighbour_point_minus & ((1 << cumulative_num_bits[d_lattice - 1]) - 1)) >> num_bits_reminder);

    }
