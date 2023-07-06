#pragma once

#include "cuda_utils.h"
#include <cuco/static_map.cuh>

typedef uint64_t KEY_TYPE;
// typedef uint32_t VALUE_TYPE;
typedef int32_t VALUE_TYPE;

// template<typename KEY_TYPE, typename VALUE_TYPE>
class hashtable_lattice {
    public:
    // rem0: [batch_size, num_positions, d_pos + 1]
    // ranks: [batch_size, num_positions, d_pos + 1]
        hashtable_lattice(
            const int _batch_size, const int _num_positions, const int _d_pos, 
            const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks);

        std::tuple<thrust::device_vector<int>, thrust::device_vector<VALUE_TYPE>> 
            get_valid_lattice_points_and_indices() const;

        void make_values_contiguous();

        std::tuple<thrust::device_vector<KEY_TYPE>, thrust::device_vector<VALUE_TYPE>> 
            get_hashtable_entries() const;

        void get_splatting_indices(
            const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output) const;

        std::vector<int> compute_all_lattice_points_slow(
            const thrust::device_ptr<int> rem0_d, const thrust::device_ptr<int> ranks_d) const;

        void compute_blur_neighbours(
            thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const;

        int get_hashtable_size() const { return hashtable_size; }

        std::vector<int> get_cumulative_num_bits() const 
        {
            std::vector<int> cumulative_num_bits_host(cumulative_num_bits_per_dim.begin(), cumulative_num_bits_per_dim.end());
            return cumulative_num_bits_host;
        }

        std::vector<int> get_min_coordinate_per_pos() const 
        {
            std::vector<int> min_coordinate_per_pos_host(min_coordinate_per_pos.begin(), min_coordinate_per_pos.end());
            return min_coordinate_per_pos_host;
        }

        size_t test_lattice_point_encoder(const thrust::device_ptr<int> rem0_d, const thrust::device_ptr<int> ranks_d) const;

    private:
        void populate_lattice_point_bits(const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks);
        void add_points_to_lattice(const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks);

        cuco::empty_key<KEY_TYPE> empty_key_sentinel{std::numeric_limits<KEY_TYPE>::max()};
        cuco::empty_value<VALUE_TYPE> empty_value_sentinel{std::numeric_limits<VALUE_TYPE>::max()};
        std::unique_ptr<cuco::static_map<KEY_TYPE, VALUE_TYPE>> hash_table;
        const int batch_size, num_positions, d_pos, d_encoded;
        int hashtable_size;
        thrust::device_vector<int> cumulative_num_bits_per_dim, min_coordinate_per_pos, max_coordinate_per_pos;
};