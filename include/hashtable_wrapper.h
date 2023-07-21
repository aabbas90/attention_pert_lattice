#pragma once

#include "hashtable_lattice.h"

class hashtable_wrapper
{
public:
    hashtable_wrapper(const int _batch_size, const int _num_positions, const int _d_pos,
                      const thrust::device_vector<int>& _min_coordinate_per_pos,
                      const thrust::device_vector<int>& _cumulative_num_bits_per_dim)
        : valid_lattice_identifier(ceil(_cumulative_num_bits_per_dim.back() / 64.0)),
          lattice(create_hashtable_lattice(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim))
    {
        if (valid_lattice_identifier < 1 || valid_lattice_identifier > 9)
            throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
    }

    int get_splatting_indices_direct(const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output)
    {
        return lattice->get_splatting_indices_direct(rem0, ranks, output);
    }

    void compute_blur_neighbours_direct(const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> splatting_table,
                                        thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const
    {
        lattice->compute_blur_neighbours_direct(rem0, ranks, splatting_table, blur_n1, blur_n2);
    }

    int get_num_lattice_points() const
    {
        return lattice->get_num_lattice_points();
    }

    std::vector<int> get_cumulative_num_bits() const
    {
        return lattice->get_cumulative_num_bits();
    }

    int encoding_size() const
    {
        return lattice->encoding_size();
    }

    std::vector<int> get_min_coordinate_per_pos() const
    {
        return lattice->get_min_coordinate_per_pos();
    }

private:
    int valid_lattice_identifier;
    std::unique_ptr<hashtable_lattice_base> lattice;

    std::unique_ptr<hashtable_lattice_base> create_hashtable_lattice(const int _batch_size, const int _num_positions, const int _d_pos,
                                                                     const thrust::device_vector<int>& _min_coordinate_per_pos,
                                                                     const thrust::device_vector<int>& _cumulative_num_bits_per_dim)
    {
        switch (valid_lattice_identifier)
        {
        case 1:
            return std::make_unique<hashtable_lattice<1>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
        case 2:
            return std::make_unique<hashtable_lattice<2>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
        case 3:
            return std::make_unique<hashtable_lattice<3>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
        case 4:
            return std::make_unique<hashtable_lattice<4>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
        case 5:
            return std::make_unique<hashtable_lattice<5>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
        case 6:
            return std::make_unique<hashtable_lattice<6>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
        case 7:
            return std::make_unique<hashtable_lattice<7>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
        case 8:
            return std::make_unique<hashtable_lattice<8>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
        case 9:
            return std::make_unique<hashtable_lattice<9>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
        default:
            throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
        }
    }
};