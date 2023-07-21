#pragma once

#include "hashtable_lattice.h"


class hashtable_wrapper
{
    public:
        hashtable_wrapper(const int _batch_size, const int _num_positions, const int _d_pos, 
            const thrust::device_vector<int>& _min_coordinate_per_pos, const thrust::device_vector<int>& _cumulative_num_bits_per_dim)
        {
            valid_lattice_identifier = ceil(_cumulative_num_bits_per_dim.back() / 64.0);
            if (valid_lattice_identifier == 1)
                hashtable_lattice_1 = std::make_unique<hashtable_lattice<1>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
            else if (valid_lattice_identifier == 2)
                hashtable_lattice_2 = std::make_unique<hashtable_lattice<2>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
            else if (valid_lattice_identifier <= 4)
                hashtable_lattice_4 = std::make_unique<hashtable_lattice<4>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
            else if (valid_lattice_identifier <= 6)
                hashtable_lattice_6 = std::make_unique<hashtable_lattice<6>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
            else if (valid_lattice_identifier <= 8)
                hashtable_lattice_8 = std::make_unique<hashtable_lattice<8>>(_batch_size, _num_positions, _d_pos, _min_coordinate_per_pos, _cumulative_num_bits_per_dim);
            else
                throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
        }
        
        void add_points_to_lattice(const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks)
        {
            if (valid_lattice_identifier == 1)
                hashtable_lattice_1->add_points_to_lattice(rem0, ranks);
            else if (valid_lattice_identifier == 2)
                hashtable_lattice_2->add_points_to_lattice(rem0, ranks);
            else if (valid_lattice_identifier <= 4)
                hashtable_lattice_4->add_points_to_lattice(rem0, ranks);
            else if (valid_lattice_identifier <= 6)
                hashtable_lattice_6->add_points_to_lattice(rem0, ranks);
            else if (valid_lattice_identifier <= 8)
                hashtable_lattice_8->add_points_to_lattice(rem0, ranks);
            else
                throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
        }

        void make_values_contiguous()
        {
            if (valid_lattice_identifier == 1)
                hashtable_lattice_1->make_values_contiguous();
            else if (valid_lattice_identifier == 2)
                hashtable_lattice_2->make_values_contiguous();
            else if (valid_lattice_identifier <= 4)
                hashtable_lattice_4->make_values_contiguous();
            else if (valid_lattice_identifier <= 6)
                hashtable_lattice_6->make_values_contiguous();
            else if (valid_lattice_identifier <= 8)
                hashtable_lattice_8->make_values_contiguous();
            else
                throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
        }

        void get_splatting_indices(
            const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output) const
        {
            if (valid_lattice_identifier == 1)
                hashtable_lattice_1->get_splatting_indices(rem0, ranks, output);
            else if (valid_lattice_identifier == 2)
                hashtable_lattice_2->get_splatting_indices(rem0, ranks, output);
            else if (valid_lattice_identifier <= 4)
                hashtable_lattice_4->get_splatting_indices(rem0, ranks, output);
            else if (valid_lattice_identifier <= 6)
                hashtable_lattice_6->get_splatting_indices(rem0, ranks, output);
            else if (valid_lattice_identifier <= 8)
                hashtable_lattice_8->get_splatting_indices(rem0, ranks, output);
            else
                throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
        }

        int get_splatting_indices_direct(
            const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output)
        {
            if (valid_lattice_identifier == 1)
                return hashtable_lattice_1->get_splatting_indices_direct(rem0, ranks, output);
            else if (valid_lattice_identifier == 2)
                return hashtable_lattice_2->get_splatting_indices_direct(rem0, ranks, output);
            else if (valid_lattice_identifier <= 4)
                return hashtable_lattice_4->get_splatting_indices_direct(rem0, ranks, output);
            else if (valid_lattice_identifier <= 6)
                return hashtable_lattice_6->get_splatting_indices_direct(rem0, ranks, output);
            else if (valid_lattice_identifier <= 8)
                return hashtable_lattice_8->get_splatting_indices_direct(rem0, ranks, output);
            else
                throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
        }

        void compute_blur_neighbours(thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const
        {
            if (valid_lattice_identifier == 1)
                hashtable_lattice_1->compute_blur_neighbours(blur_n1, blur_n2);
            else if (valid_lattice_identifier == 2)
                hashtable_lattice_2->compute_blur_neighbours(blur_n1, blur_n2);
            else if (valid_lattice_identifier <= 4)
                hashtable_lattice_4->compute_blur_neighbours(blur_n1, blur_n2);
            else if (valid_lattice_identifier <= 6)
                hashtable_lattice_6->compute_blur_neighbours(blur_n1, blur_n2);
            else if (valid_lattice_identifier <= 8)
                hashtable_lattice_8->compute_blur_neighbours(blur_n1, blur_n2);
            else
                throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
        }

        void compute_blur_neighbours_direct(const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> rank, thrust::device_ptr<VALUE_TYPE> splatting_table, 
            thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const
        {
            if (valid_lattice_identifier == 1)
                hashtable_lattice_1->compute_blur_neighbours_direct(rem0, rank, splatting_table, blur_n1, blur_n2);
            else if (valid_lattice_identifier == 2)
                hashtable_lattice_2->compute_blur_neighbours_direct(rem0, rank, splatting_table, blur_n1, blur_n2);
            else if (valid_lattice_identifier <= 4)
                hashtable_lattice_4->compute_blur_neighbours_direct(rem0, rank, splatting_table, blur_n1, blur_n2);
            else if (valid_lattice_identifier <= 6)
                hashtable_lattice_6->compute_blur_neighbours_direct(rem0, rank, splatting_table, blur_n1, blur_n2);
            else if (valid_lattice_identifier <= 8)
                hashtable_lattice_8->compute_blur_neighbours_direct(rem0, rank, splatting_table, blur_n1, blur_n2);
            else
                throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
        }

        int get_num_lattice_points() const 
        {
            if (valid_lattice_identifier == 1)
                return hashtable_lattice_1->get_num_lattice_points();
            else if (valid_lattice_identifier == 2)
                return hashtable_lattice_2->get_num_lattice_points();
            else if (valid_lattice_identifier <= 4)
                return hashtable_lattice_4->get_num_lattice_points();
            else if (valid_lattice_identifier <= 6)
                return hashtable_lattice_6->get_num_lattice_points();
            else if (valid_lattice_identifier <= 8)
                return hashtable_lattice_8->get_num_lattice_points();
            else
                throw std::runtime_error("Unsupported encoding size = " + std::to_string(valid_lattice_identifier));
        }

    private:
        int valid_lattice_identifier;
        std::unique_ptr<hashtable_lattice<1>> hashtable_lattice_1;
        std::unique_ptr<hashtable_lattice<2>> hashtable_lattice_2;
        std::unique_ptr<hashtable_lattice<4>> hashtable_lattice_4;
        std::unique_ptr<hashtable_lattice<6>> hashtable_lattice_6;
        std::unique_ptr<hashtable_lattice<8>> hashtable_lattice_8;
};