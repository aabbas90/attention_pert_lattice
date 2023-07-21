#include "cuda_utils.h"
#include "time_measure_util.h"
#include "attention_pert_lattice.h"
#include <thrust/extrema.h>
#include <thrust/for_each.h>

template<typename REAL>
struct put_ranks_rem0 {
    const size_t num_positions;
    const int d_lattice;
    const int* feat_v_sum;
    const int* const sorting_order;
    const REAL* const features;
    int* ranks;
    int* rem0; 
    REAL* y;
    __host__ __device__ void operator()(const size_t i)
    {
        // TODOAA: Use shared memory to write ranks?
        const int out_index = sorting_order[i];
        int cur_rank = i % d_lattice + feat_v_sum[i / d_lattice];
        const int add_term = cur_rank < 0 ? d_lattice: 0;
        const int minus_term = cur_rank >= d_lattice ? -d_lattice: 0;
        ranks[out_index] = cur_rank + feat_v_sum[i / d_lattice] + add_term + minus_term;
        const int new_rem0 = rem0[out_index] + add_term + minus_term;
        rem0[out_index] = new_rem0;
        y[out_index] = (features[out_index] - new_rem0) / d_lattice;
    }
};

template<typename REAL>
struct compute_barycentric {
    const int d_lattice;
    const REAL* const y;
    REAL* barycentric;
    __host__ __device__ void operator()(const size_t i)
    {
        const int index_pt = i / d_lattice;
        const int index_d = i % d_lattice;
        if (index_d == 0)
            barycentric[i] -= y[index_pt * d_lattice + d_lattice - 1] - 1;
        else
            barycentric[i] -= y[index_pt * d_lattice + (index_d - 1)];
    }
};

//     pos_features: [num_positions, d_lattice]
//     rem0: [num_positions, d_lattice]
//     rank: [num_positions, d_lattice]
//     barycentric: [num_positions, d_lattice].
template<typename REAL>
void compute_rem0_rank_barycentric(const size_t num_positions, const int d_lattice,
        const thrust::device_ptr<REAL> features, thrust::device_ptr<int> rem0, thrust::device_ptr<int> ranks,
        thrust::device_ptr<REAL> barycentric)
{
    thrust::device_vector<int> feat_v(num_positions * d_lattice);
    thrust::device_vector<REAL> feat_minus_rem0(num_positions * d_lattice);
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("[compute_rem0_rank_barycentric] round")
        auto first_output = thrust::make_zip_iterator(
            thrust::make_tuple(feat_v.begin(), rem0, feat_minus_rem0.begin()));

        thrust::transform(features, features + num_positions * d_lattice, first_output, 
            [d_lattice] __device__(auto v) {
                const int rounded = round(v / d_lattice);
                const REAL rem0 = rounded * d_lattice;
                return thrust::make_tuple(rounded, rem0, v - rem0); 
            });
    }
    {   
        // e.g. feat_minus_rem0 = [[5, 2, 3], [1, 8, 4]]
        // sorted = [[5, 3, 2], [8, 4, 1]], sorted indices = [[0, 2, 1], [4, 5, 3]]
        // [[0, 1, 2], [0, 1, 2]], 4th location has rank 0 ...
        // ranks = [[0, 2, 1], [5, 3, 4]]
        thrust::device_vector<int> sorting_order(feat_minus_rem0.size());
        thrust::sequence(sorting_order.begin(), sorting_order.end());
         
        auto get_point_index = [d_lattice] __device__ (const auto i) {return i / d_lattice; };
        auto first_input = thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), get_point_index),
            feat_minus_rem0.begin()));

        auto last_input = thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0) + feat_minus_rem0.size(), 
                    get_point_index),
            feat_minus_rem0.end()));
        
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("[compute_rem0_rank_barycentric] sorting for rank")
            thrust::stable_sort_by_key(
                first_input, last_input, sorting_order.begin(), thrust::greater<thrust::tuple<int, REAL>>());
        }

        thrust::device_vector<int> feat_v_sum(num_positions);
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("[compute_rem0_rank_barycentric] reduce_by_key")
            thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), 
                    get_point_index),
                thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0) + feat_minus_rem0.size(), 
                    get_point_index),
                feat_v.begin(), thrust::make_discard_iterator(), feat_v_sum.begin());
        }
        thrust::device_vector<REAL> y(num_positions * d_lattice);
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("[compute_rem0_rank_barycentric] put_ranks_rem0")

            thrust::for_each(
                thrust::make_counting_iterator<int>(0), 
                thrust::make_counting_iterator<int>(0) + y.size(),
                put_ranks_rem0<REAL>({num_positions, d_lattice, thrust::raw_pointer_cast(feat_v_sum.data()),
                    thrust::raw_pointer_cast(sorting_order.data()), thrust::raw_pointer_cast(features),
                    thrust::raw_pointer_cast(ranks), thrust::raw_pointer_cast(rem0),
                    thrust::raw_pointer_cast(y.data())}));
        }
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("[compute_rem0_rank_barycentric] sort barycentric")

            thrust::stable_sort(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), get_point_index),
                    y.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0) + y.size(), get_point_index),
                    y.end())));
        }
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("[compute_rem0_rank_barycentric] compute_barycentric")
            thrust::copy(y.begin(), y.end(), barycentric);
            thrust::for_each(
                thrust::make_counting_iterator<int>(0), 
                thrust::make_counting_iterator<int>(0) + y.size(),
                compute_barycentric<REAL>({d_lattice, 
                    thrust::raw_pointer_cast(y.data()),
                    thrust::raw_pointer_cast(barycentric)}));
        }
    }
}

template<typename REAL>
struct splat_func {
    const size_t num_positions;
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
        assert(index_pt < num_positions);

        const REAL cur_value = values[index_pt * d_value + index_d];
        
        const size_t u = index_pt * num_neighbours + index_n;
        const int cur_splat_index = out_indices[u];
        const REAL cur_weight = barycentric[u];

        assert(cur_splat_index < num_splatted_points);
        const size_t out_index = cur_splat_index * d_value + index_d;
        atomicAdd(&splatted_output[out_index], cur_value * cur_weight);
    }
};

//     values: [num_positions, d_value]
//     out_indices: [num_positions, num_neighbours]
//     barycentric: [num_positions, num_neighbours]
//     splatted_output: [num_splatted_points, d_value] and initialized by 0.
template<typename REAL>
void splat_lattice(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<REAL> values, const thrust::device_ptr<int> out_indices, const thrust::device_ptr<REAL> barycentric,
        thrust::device_ptr<REAL> splatted_output)
{
    assert(*thrust::max_element(out_indices, out_indices + (num_positions * num_neighbours)) < num_splatted_points);
    const size_t num_workers = num_positions * d_value * num_neighbours;

    splat_func<REAL> func({num_positions, num_neighbours, d_value, num_splatted_points,
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
        thrust::device_ptr<REAL> splatted_values, const thrust::device_ptr<int> blur_n1, const thrust::device_ptr<int> blur_n2, const bool do_reverse)
{
    const size_t num_workers = num_lattice_points * d_value;
    thrust::device_vector<REAL> temp(num_lattice_points * d_value);
    thrust::device_ptr<REAL> in_ptr = splatted_values;
    thrust::device_ptr<REAL> out_ptr = temp.data();
    // std::cout<<"num_directions: "<<num_directions<<", num_lattice_points: "<<num_lattice_points<<", d_value: "<<d_value<<", filter_coeff_self: "<<filter_coeff_self<<", filter_coeff_n1: "<<filter_coeff_n1<<"\n";
    if (do_reverse) {
        for (int blur_d = num_directions - 1; blur_d >= 0; blur_d--)
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
    }
    else {
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
    }
    thrust::swap(in_ptr, out_ptr);
    if (num_directions % 2 == 1) // For odd many steps, temp contains the final result so copy it to input array.
        thrust::copy(temp.begin(), temp.end(), splatted_values);
}

template<typename REAL>
struct slice_func {
    const size_t num_positions;
    const size_t num_neighbours; 
    const size_t d_value;
    const size_t num_splatted_points;
    const REAL* const splatted_input; 
    const int* const out_indices;
    const REAL* const barycentric;
    REAL* sliced_values;
    // consecutive threads operate on different channels of same sliced point.
    __host__ __device__ void operator()(const size_t i)
    {
        const size_t index_d = i % d_value;
        const size_t index_pt = i / d_value;
        assert(index_pt < num_positions);
        REAL cur_sliced = 0;
        for(int index_n = 0; index_n != num_neighbours; ++index_n)
        {
            const size_t u = index_pt * num_neighbours + index_n;
            const int cur_splat_index = out_indices[u];
            const REAL cur_weight = barycentric[u];

            assert(cur_splat_index < num_splatted_points);
            const int in_index = cur_splat_index * d_value + index_d;
            cur_sliced += cur_weight * splatted_input[in_index];
        }
        // sliced_values[index_pt * d_value + index_d] = cur_sliced;
        sliced_values[i] = cur_sliced;
    }
};

//     splatted: [num_splatted_points, d_value]
//     in_indices: [num_positions, num_neighbours]
//     barycentric: [num_positions, num_neighbours]
//     sliced_output: [num_positions, d_value]
template<typename REAL>
void slice_lattice(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<REAL> splatted, const thrust::device_ptr<int> in_indices, const thrust::device_ptr<REAL> barycentric,
        thrust::device_ptr<REAL> sliced_output)
{
    assert(*thrust::max_element(in_indices, in_indices + (num_positions * num_neighbours)) < num_splatted_points);
    const size_t num_workers = num_positions * d_value;

    slice_func<REAL> func({num_positions, num_neighbours, d_value, num_splatted_points,
                        thrust::raw_pointer_cast(splatted),
                        thrust::raw_pointer_cast(in_indices),
                        thrust::raw_pointer_cast(barycentric),
                        thrust::raw_pointer_cast(sliced_output)});
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, func);
}

template<typename REAL>
struct grad_barycentric_func {
    const size_t num_positions;
    const size_t num_neighbours; 
    const size_t d_value;
    const size_t num_splatted_points;
    const REAL* const splatted_input; 
    const int* const out_indices;
    const REAL* const values;
    REAL* result;
    // consecutive threads operate on different channels of same sliced point.
    __host__ __device__ void operator()(const size_t i)
    {
        const size_t index_n = i % num_neighbours;
        const size_t index_pt = i / num_neighbours;
        assert(index_pt < num_positions);
        REAL cur_sliced = 0;
        const int cur_splat_index = out_indices[index_pt * num_neighbours + index_n];
        assert(cur_splat_index < num_splatted_points);
        const int pt_offset = index_pt * d_value;
        const int sp_offset = cur_splat_index * d_value;
        for(int index_d = 0; index_d != d_value; ++index_d)
            cur_sliced += values[pt_offset + index_d] * splatted_input[sp_offset + index_d];
        result[i] = cur_sliced;
    }
};

//     splatted: [num_splatted_points, d_value]
//     in_indices: [num_positions, num_neighbours]
//     values: [num_positions, d_value]
//     sliced_output: [num_positions, num_neighbours]
template<typename REAL>
void grad_barycentric(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<REAL> splatted, const thrust::device_ptr<int> in_indices, const thrust::device_ptr<REAL> values,
        thrust::device_ptr<REAL> sliced_output)
{
    assert(*thrust::max_element(in_indices, in_indices + (num_positions * num_neighbours)) < num_splatted_points);
    const size_t num_workers = num_positions * num_neighbours;

    grad_barycentric_func<REAL> func({num_positions, num_neighbours, d_value, num_splatted_points,
                        thrust::raw_pointer_cast(splatted),
                        thrust::raw_pointer_cast(in_indices),
                        thrust::raw_pointer_cast(values),
                        thrust::raw_pointer_cast(sliced_output)});
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, func);
}

template void compute_rem0_rank_barycentric(const size_t num_positions, const int d_lattice,
        const thrust::device_ptr<float> features, const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks,
        thrust::device_ptr<float> barycentric);

template void splat_lattice(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
    const thrust::device_ptr<float> values, const thrust::device_ptr<int> out_indices, const thrust::device_ptr<float> barycentric,
    thrust::device_ptr<float> splatted_output);

// template void splat_lattice(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
//     const thrust::device_ptr<double> values, const thrust::device_ptr<int> out_indices, const thrust::device_ptr<double> barycentric,
//     thrust::device_ptr<double> splatted_output);

template void blur_lattice(const size_t num_lattice_points, const size_t num_directions, const size_t d_value, 
        const int invalid_blur_neighbour_value, const float filter_coeff_self, const float filter_coeff_n1, 
        thrust::device_ptr<float> splatted_values, const thrust::device_ptr<int> blur_n1, const thrust::device_ptr<int> blur_n2, const bool do_reverse);

template void slice_lattice(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<float> splatted, const thrust::device_ptr<int> in_indices, const thrust::device_ptr<float> barycentric,
        thrust::device_ptr<float> sliced_output);

template void grad_barycentric(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<float> splatted, const thrust::device_ptr<int> in_indices, const thrust::device_ptr<float> values,
        thrust::device_ptr<float> sliced_output);