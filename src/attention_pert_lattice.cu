#include "cuda_utils.h"
#include "time_measure_util.h"
#include "attention_pert_lattice.h"
#include <thrust/extrema.h>
#include <thrust/for_each.h>

template<typename REAL>
struct splat_func {
    const size_t num_points;
    const size_t num_neighbours; 
    const size_t d_value;
    const size_t num_splatted_points;
    const REAL* const values;
    const int* const out_indices;
    const REAL* const barycentric;
    REAL* splatted_output; 
    __host__ __device__ void operator()(const size_t i)
    {
        const size_t index_d = i % d_value;
        const size_t index_n = (i / d_value) % num_neighbours;
        const size_t index_pt = (i / (d_value * num_neighbours));
        assert(index_pt < num_points);

        const REAL cur_value = values[index_pt * d_value + index_d];
        
        const size_t u = index_pt * num_neighbours + index_n;
        const int cur_splat_index = out_indices[u];
        const REAL cur_weight = barycentric[u];

        assert(cur_splat_index < num_splatted_points);
        const size_t out_index = cur_splat_index * d_value + index_d;
        atomicAdd(&splatted_output[out_index], cur_value * cur_weight);
    }
};

//     values: [num_points, d_value]
//     out_indices: [num_points, num_neighbours]
//     barycentric: [num_points, num_neighbours]
//     splatted_output: [num_splatted_points, d_value] and initialized by 0.
template<typename REAL>
void splat_lattice(const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<REAL> values, const thrust::device_ptr<int> out_indices, const thrust::device_ptr<REAL> barycentric,
        thrust::device_ptr<REAL> splatted_output)
{
    assert(*thrust::max_element(out_indices, out_indices + (num_points * num_neighbours)) < num_splatted_points);
    const size_t num_workers = num_points * d_value * num_neighbours;

    splat_func<REAL> func({num_points, num_neighbours, d_value, num_splatted_points,
                        thrust::raw_pointer_cast(values),
                        thrust::raw_pointer_cast(out_indices),
                        thrust::raw_pointer_cast(barycentric),
                        thrust::raw_pointer_cast(splatted_output)});
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, func);
}

template<typename REAL>
struct blur_func {
    const int blur_direction;
    const size_t num_lattice_points;
    const size_t d_value;
    const int invalid_neighbour_value;
    const REAL self_coeff;
    const REAL n1_coeff;
    const REAL* const splatted_values;
    const int* const blur_n1;
    const int* const blur_n2;
    REAL* blurred_values;
    __host__ __device__ void operator()(const size_t i)
    {
        const int index_d = i % d_value;
        const int index_pt = i / d_value;

        assert(index_pt < num_lattice_points);

        const int self_index = index_pt * d_value + index_d;
        const int index_n1 = blur_n1[blur_direction * num_lattice_points + index_pt];
        const int index_n2 = blur_n2[blur_direction * num_lattice_points + index_pt];

        // if (index_n1 < 0 && index_n2 < 0)
        //     return;
        REAL filter_sum = self_coeff;
        REAL out = self_coeff * splatted_values[self_index];
        if (index_n1 != invalid_neighbour_value)
        {
            out += n1_coeff * splatted_values[index_n1 * d_value + index_d];
            filter_sum += n1_coeff;
        }
        if (index_n2 != invalid_neighbour_value)
        {
            out += n1_coeff * splatted_values[index_n2 * d_value + index_d];
            filter_sum += n1_coeff;
        }            
        blurred_values[self_index] = out / filter_sum;
        // printf("self_value: %f, out_value: %f, n1_value: %f, n2_value: %f, index_pt: %d, index_d: %d, self_index: %d, index_n1: %d, index_n2: %d, flat_n1: %d, flat_n2: %d\n", 
        //     splatted_values[self_index], out, splatted_values[index_n1 * d_value + index_d], splatted_values[index_n2 * d_value + index_d],
        //     index_pt, index_d, self_index, index_n1, index_n2, index_n1 * d_value + index_d, index_n2 * d_value + index_d);
    }
};

//     blur_n1: [num_directions, num_splatted_points]
//     blur_n2: [num_directions, num_splatted_points]
//     splatted_values: [num_splatted_points, d_value], will also contains output.
template<typename REAL>
void blur_lattice(const size_t num_lattice_points, const size_t num_directions, const size_t d_value, const int invalid_neighbour_value,
        const REAL filter_coeff_self, const REAL filter_coeff_n1,
        thrust::device_ptr<REAL> splatted_values, const thrust::device_ptr<int> blur_n1, const thrust::device_ptr<int> blur_n2)
{
    const size_t num_workers = num_lattice_points * d_value;
    thrust::device_vector<REAL> temp(num_lattice_points * d_value);
    thrust::device_ptr<REAL> in_ptr = splatted_values;
    thrust::device_ptr<REAL> out_ptr = temp.data();
    std::cout<<"num_directions: "<<num_directions<<", num_lattice_points: "<<num_lattice_points<<", d_value: "<<d_value<<", filter_coeff_self: "<<filter_coeff_self<<", filter_coeff_n1: "<<filter_coeff_n1<<"\n";
    for (int blur_d = 0; blur_d < num_directions; blur_d++)
    {
        blur_func<REAL> func({blur_d, num_lattice_points, d_value, invalid_neighbour_value,
                            filter_coeff_self, filter_coeff_n1,
                            thrust::raw_pointer_cast(in_ptr),
                            thrust::raw_pointer_cast(blur_n1),
                            thrust::raw_pointer_cast(blur_n2),
                            thrust::raw_pointer_cast(out_ptr)});
        thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, func);
        thrust::swap(in_ptr, out_ptr);
    }
    thrust::swap(in_ptr, out_ptr);
    if (num_directions % 2 == 1) // For odd many steps, temp contains the final result so copy it to input array.
        thrust::copy(temp.begin(), temp.end(), splatted_values);
}


template void splat_lattice(const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
    const thrust::device_ptr<float> values, const thrust::device_ptr<int> out_indices, const thrust::device_ptr<float> barycentric,
    thrust::device_ptr<float> splatted_output);

// template void splat_lattice(const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
//     const thrust::device_ptr<double> values, const thrust::device_ptr<int> out_indices, const thrust::device_ptr<double> barycentric,
//     thrust::device_ptr<double> splatted_output);

template void blur_lattice(const size_t num_lattice_points, const size_t num_directions, const size_t d_value, 
        const int invalid_blur_neighbour_value, const float filter_coeff_self, const float filter_coeff_n1, 
        thrust::device_ptr<float> splatted_values, const thrust::device_ptr<int> blur_n1, const thrust::device_ptr<int> blur_n2);
