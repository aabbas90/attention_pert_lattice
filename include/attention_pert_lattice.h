#pragma once

#include "cuda_utils.h"

template<typename REAL>
void splat_lattice(const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<REAL> values, const thrust::device_ptr<int> out_indices, const thrust::device_ptr<REAL> barycentric,
        thrust::device_ptr<REAL> splatted_output);

template<typename REAL>
void blur_lattice(const size_t num_lattice_points, const size_t num_directions, const size_t d_value, const int invalid_blur_neighbour_value,
        const REAL filter_coeff_self, const REAL filter_coeff_n1,
        thrust::device_ptr<REAL> splatted_values, const thrust::device_ptr<int> blur_n_1, const thrust::device_ptr<int> blur_n_2);

void compute_neighbours_and_indices(
        const size_t batch_size, const size_t num_points, const size_t d, const size_t filter_radius,
        const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, 
        const thrust::device_ptr<int> out_blur_n, const thrust::device_ptr<int> out_indices);