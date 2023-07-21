#pragma once

#include "cuda_utils.h"

template<typename REAL>
void compute_rem0_rank_barycentric(const size_t num_positions, const int d_lattice,
        const thrust::device_ptr<REAL> features, thrust::device_ptr<int> rem0, thrust::device_ptr<int> ranks,
        thrust::device_ptr<REAL> barycentric);

template<typename REAL>
void splat_lattice(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<REAL> values, const thrust::device_ptr<int> out_indices, const thrust::device_ptr<REAL> barycentric,
        thrust::device_ptr<REAL> splatted_output);

template<typename REAL>
void blur_lattice(const size_t num_lattice_points, const size_t num_directions, const size_t d_value, const int invalid_blur_neighbour_value,
        const REAL filter_coeff_self, const REAL filter_coeff_n1,
        thrust::device_ptr<REAL> splatted_values, const thrust::device_ptr<int> blur_n_1, const thrust::device_ptr<int> blur_n_2, const bool do_reverse);

template<typename REAL>
void slice_lattice(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<REAL> splatted, const thrust::device_ptr<int> in_indices, const thrust::device_ptr<REAL> barycentric,
        thrust::device_ptr<REAL> sliced_output);

template<typename REAL>
void grad_barycentric(const size_t num_positions, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const thrust::device_ptr<REAL> splatted, const thrust::device_ptr<int> in_indices, const thrust::device_ptr<REAL> barycentric,
        thrust::device_ptr<REAL> sliced_output);
