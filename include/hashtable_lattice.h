#pragma once

#include "cuda_utils.h"
#include <cuco/static_map.cuh>

typedef int VALUE_TYPE;

// User-defined device hasher
template<int NUM_VECS>
struct hash_encoded_lattice_pt {
    __host__ __device__ uint64_t operator()(const encoded_lattice_pt<NUM_VECS>& k) const
    {
        cuco::murmurhash3_fmix_64<uint64_t> hash;
        uint64_t hashcode = 0;
        for (int i = 0; i != NUM_VECS; ++i)
            hashcode ^= hash(k.data[i]);
        return hashcode;
    }
};

// rem0: [batch_size, num_positions, d_pos + 1]
// ranks: [batch_size, num_positions, d_pos + 1]
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> calculate_lattice_extents(
    const int batch_size, const int num_positions, const int d_pos, 
    const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks);

template<int NUM_VECS>
class hashtable_lattice {
    public:
        hashtable_lattice() : batch_size(0), num_positions(0), d_pos(0), d_encoded(0), num_lattice_points(0) {}

        hashtable_lattice(
            const int _batch_size, const int _num_positions, const int _d_pos, 
            const thrust::device_vector<int>& _min_coordinate_per_pos, const thrust::device_vector<int>& _cumulative_num_bits_per_dim);

        void add_points_to_lattice(const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks);

        std::tuple<thrust::device_vector<int>, thrust::device_vector<encoded_lattice_pt<NUM_VECS>>, thrust::device_vector<VALUE_TYPE>> 
            get_valid_lattice_points_and_indices() const;

        void make_values_contiguous();

        std::tuple<thrust::device_vector<encoded_lattice_pt<NUM_VECS>>, thrust::device_vector<VALUE_TYPE>> 
            get_hashtable_entries() const;

        void get_splatting_indices(
            const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output) const;

        int get_splatting_indices_direct(
            const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output);

        void compute_blur_neighbours(
            thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const;

        void compute_blur_neighbours_direct(
            const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> splatting_table, 
            thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const;

        // std::vector<int> compute_all_lattice_points_slow(
        //     const thrust::device_ptr<int> rem0_d, const thrust::device_ptr<int> ranks_d) const;

        int get_num_lattice_points() const { return num_lattice_points; }

        std::vector<int> get_cumulative_num_bits() const 
        {
            std::vector<int> cumulative_num_bits_host(cumulative_num_bits_per_dim.begin(), cumulative_num_bits_per_dim.end());
            return cumulative_num_bits_host;
        }

        int encoding_size() const { return ceil(number_of_required_bits / 64.0); }

        std::vector<int> get_min_coordinate_per_pos() const 
        {
            std::vector<int> min_coordinate_per_pos_host(min_coordinate_per_pos.begin(), min_coordinate_per_pos.end());
            return min_coordinate_per_pos_host;
        }

        // size_t test_lattice_point_encoder(const thrust::device_ptr<int> rem0_d, const thrust::device_ptr<int> ranks_d) const;

    private:
        cuco::empty_value<VALUE_TYPE> empty_value_sentinel { -1 };
        std::unique_ptr<cuco::static_map<encoded_lattice_pt<NUM_VECS>, VALUE_TYPE>> hash_table;
        const int batch_size, num_positions, d_pos, d_encoded;
        int num_lattice_points;
        int number_of_required_bits;
        thrust::device_vector<int> cumulative_num_bits_per_dim, min_coordinate_per_pos;
};