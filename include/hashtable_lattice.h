#pragma once

#include "cuda_utils.h"

typedef int VALUE_TYPE;

// rem0: [batch_size, num_positions, d_pos + 1]
// ranks: [batch_size, num_positions, d_pos + 1]
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> calculate_lattice_extents(
    const int batch_size, const int num_positions, const int d_pos, 
    const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks);

class hashtable_lattice_base
{
public:
    virtual ~hashtable_lattice_base() = default;

    virtual int get_splatting_indices_direct(
        const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output) = 0;

    virtual void compute_blur_neighbours_direct(
        const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> splatting_table,
        thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const = 0;

    virtual int get_num_lattice_points() const = 0;

    virtual std::vector<int> get_cumulative_num_bits() const = 0;

    virtual int encoding_size() const = 0;

    virtual std::vector<int> get_min_coordinate_per_pos() const = 0;
};

template <int NUM_VECS>
class hashtable_lattice : public hashtable_lattice_base {
public:
    hashtable_lattice() : batch_size(0), num_positions(0), d_pos(0), d_encoded(0), num_lattice_points(0) {}

    hashtable_lattice(
        const int _batch_size, const int _num_positions, const int _d_pos,
        const thrust::device_vector<int>& _min_coordinate_per_pos, const thrust::device_vector<int>& _cumulative_num_bits_per_dim);

    int get_splatting_indices_direct(
        const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output) override;

    void compute_blur_neighbours_direct(
        const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> splatting_table,
        thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const override;

    int get_num_lattice_points() const override { return num_lattice_points; }

    std::vector<int> get_cumulative_num_bits() const override
    {
        std::vector<int> cumulative_num_bits_host(cumulative_num_bits_per_dim.begin(), cumulative_num_bits_per_dim.end());
        return cumulative_num_bits_host;
    }

    int encoding_size() const override { return ceil(number_of_required_bits / 64.0); }

    std::vector<int> get_min_coordinate_per_pos() const override
    {
        std::vector<int> min_coordinate_per_pos_host(min_coordinate_per_pos.begin(), min_coordinate_per_pos.end());
        return min_coordinate_per_pos_host;
    }

private:
    const int batch_size, num_positions, d_pos, d_encoded;
    int num_lattice_points;
    int number_of_required_bits;
    thrust::device_vector<int> cumulative_num_bits_per_dim, min_coordinate_per_pos;
};