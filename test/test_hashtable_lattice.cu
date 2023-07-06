#include <vector>
#include <thrust/device_vector.h>
#include <map>
#include "hashtable_lattice.h"
#include "cuda_utils.h"

void check_rows(const std::vector<int>& reference, const std::vector<int>& computed, const int num_columns)
{
    // Calculate the size of each row based on the number of columns
    const int row_size = num_columns;

    // Iterate over the computed vector, checking if each row is present in the reference vector
    for (int i = 0; i < computed.size(); i += row_size) {
        // Get the subrange representing the current row in the computed vector
        auto computed_row_begin = computed.begin() + i;
        auto computed_row_end = computed_row_begin + row_size;

        // Check if the current row in the computed vector exists in the reference vector
        auto row_present = std::search(reference.begin(), reference.end(), computed_row_begin, computed_row_end);

        // If the row is not found in the reference vector, output a message
        if (row_present == reference.end()) {
            std::cout << "\t Row not found: ";
            for (auto it = computed_row_begin; it != computed_row_end; ++it) {
                std::cout << *it << " ";
            }
            std::cout << std::endl;
        }
    }
}

bool check_blur_neighbours(const thrust::device_vector<KEY_TYPE>& keys_d, const thrust::device_vector<VALUE_TYPE>& values_d, 
    const thrust::device_vector<int>& blur_neighbours_d, const std::vector<int>& cumulative_num_bits, 
    const std::vector<int>& min_coordinate, const int d_pos, const bool is_positive_direction)
{
    auto print_vector = [](const std::vector<int>& vec, const char* name) {
        std::cout<<name<<": ";
        for (const auto& element : vec) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    };
    bool failed = false;
    const int num_pts = keys_d.size();
    std::vector<KEY_TYPE> keys(keys_d.begin(), keys_d.end());
    std::vector<VALUE_TYPE> values(values_d.begin(), values_d.end());

    std::map<KEY_TYPE, VALUE_TYPE> map;
    std::map<VALUE_TYPE, KEY_TYPE> map_inverse;
    for(int i = 0; i != num_pts; ++i)
    {
        map.emplace(keys[i], values[i]);
        if (values[i] < num_pts)
            map_inverse.emplace(values[i], keys[i]);
    }

    std::vector<int> blur_neighbours(blur_neighbours_d.begin(), blur_neighbours_d.end());
    auto add_vectors = [] (std::vector<int> v1, std::vector<int> v2)
    {
        assert (v1.size() == v2.size());
        std::vector<int> v3(v1.size());
        std::transform(v1.begin(), v1.end(), v2.begin(), v3.begin(), std::plus<int>());
        return v3;
    };
    for (auto const& [encoded_point, self_index]: map)
    {
        std::vector<int> self_point(d_pos + 1);
        decode_point(cumulative_num_bits.data(), min_coordinate.data(), 0, d_pos + 1, encoded_point, self_point.data());
        for(int direction = 0; direction < d_pos + 1; direction++)
        {
            const int neighbour_index = blur_neighbours[direction * num_pts + self_index];
            if (neighbour_index >= num_pts) // neighbour not present.
                continue;
            std::vector<int> req_offset(d_pos + 1, is_positive_direction ? -1 : 1);
            req_offset[0] = 0; // batch index.
            req_offset[direction + 1] = is_positive_direction ? d_pos : -d_pos;
            std::vector<int> req_neighbour = add_vectors(self_point, req_offset);

            KEY_TYPE neighbour_encoded = map_inverse[neighbour_index];
            std::vector<int> neighbour_point(d_pos + 1);
            decode_point(cumulative_num_bits.data(), min_coordinate.data(), 0, d_pos + 1, neighbour_encoded, neighbour_point.data());
            std::vector<int> computed_offset(d_pos + 1);
            std::transform(neighbour_point.begin(), neighbour_point.end(), self_point.begin(), computed_offset.begin(), std::minus<int>());
            if (computed_offset != req_offset)
            {
                std::cout<<"\n";
                std::cout<<"index: "<<self_index<<", direction: "<<direction<<", neighbour_index: "<<neighbour_index<<"\n";
                std::cout<<"self_encoded: "<<encoded_point<<"\n";
                std::cout<<"neighbour_encoded: "<<neighbour_encoded<<"\n";
                print_vector(self_point, "self_point");
                print_vector(neighbour_point, "neighbour_point");
                print_vector(req_offset, "required_offset");
                print_vector(computed_offset, "computed_offset");
                failed = true;
            }
        }
    }
    return failed;
}

int main(int argc, char** argv)
{
    const int batch_size = 2;
    const int d_pos = 3;
    const int num_positions = 4;

    std::vector<int> rem0_h(32, 0);
    thrust::device_vector<int> rem0(rem0_h);
    std::vector<int> ranks_h = {
        1, 0, 3, 2,
        3, 2, 0, 1,
        1, 3, 0, 2,
        3, 1, 0, 2,
        2, 1, 0, 3,
        1, 2, 0, 3,
        3, 0, 1, 2,
        0, 1, 2, 3};
    thrust::device_vector<int> ranks(ranks_h);

    std::vector<int> lattice_points_h = {
        0,  0,  0,  0,  
        0, -1,  3, -1,  
        0, -2, -2,  2,  
        0, -3,  1,  1,  
        0,  2, -2,  2,  
        0, -1, -1,  3,  
        0, -2,  2,  2,  
        0,  1,  1, -3,  
        0,  2,  2, -2,  
        0,  1, -3,  1,
        1, -3,  1,  1,  
        1,  2, -2,  2,  
        1,  1,  1,  1,  
        1, -1,  3, -1,  
        1, -1, -1,  3,  
        1, -2,  2,  2,  
        1,  3, -1, -1,  
        1,  2,  2, -2,  
        1,  0,  0,  0};

    hashtable_lattice htl(batch_size, num_positions, d_pos, rem0.data(), ranks.data());
    {
        const auto out = htl.get_valid_lattice_points_and_indices();
        thrust::device_vector<int> lattice_points = std::get<0>(out);
        std::vector<int> lattice_points_computed_h(lattice_points.begin(), lattice_points.end());
        thrust::device_vector<int> indices = std::get<1>(out);
        print_matrix(lattice_points, "lattice_points", d_pos + 1);
        print_vector(indices, "indices");
        std::cout<<"Checking which computed key is not in reference.\n";
        check_rows(lattice_points_h, lattice_points_computed_h, d_pos + 1);
        std::cout<<"Checking which reference key is not in computed.\n";
        check_rows(lattice_points_computed_h, lattice_points_h, d_pos + 1);

        // std::vector<int> lattice_pts_exhaustive = htl.compute_all_lattice_points_slow(rem0.data(), ranks.data());    
        // check_rows(lattice_points_h, lattice_pts_exhaustive, d_pos + 1);
    }

    // {
    //     thrust::device_vector<VALUE_TYPE> splatting_indices(batch_size * num_positions * (d_pos + 1));
    //     htl.get_splatting_indices(rem0.data(), ranks.data(), splatting_indices.data());
    //     print_matrix(splatting_indices, "splatting_indices", d_pos + 1);
    // }
    {
        thrust::device_vector<VALUE_TYPE> blur_n1((d_pos + 1) * htl.get_hashtable_size());
        thrust::device_vector<VALUE_TYPE> blur_n2((d_pos + 1) * htl.get_hashtable_size());
        htl.compute_blur_neighbours(blur_n1.data(), blur_n2.data());
        thrust::device_vector<KEY_TYPE> keys;
        thrust::device_vector<VALUE_TYPE> values;
        std::tie(keys, values) = htl.get_hashtable_entries();
        std::vector<int> cumulative_num_bits = htl.get_cumulative_num_bits();
        std::vector<int> min_coordinate = htl.get_min_coordinate_per_pos();
        if(check_blur_neighbours(keys, values, blur_n1, cumulative_num_bits, min_coordinate, d_pos, true))
            std::cout<<"\tBlur neighbour test in positive direction failed.\n";
        if(check_blur_neighbours(keys, values, blur_n2, cumulative_num_bits, min_coordinate, d_pos, false))
            std::cout<<"\tBlur neighbour test in negative direction failed.\n";
    }
    // const std::vector<int> cumulative_num_bits_host = htl.get_cumulative_num_bits();

    // KEY_TYPE neighbour_point_plus, neighbour_point_minus;
    // bool plus_overflow, minus_overflow;
    // const KEY_TYPE encoded_point = 69;
    // for (int direction = 2; direction != 3; direction++)
    // {
    //     compute_neighbour_encoding(cumulative_num_bits_host.data(), d_pos + 1, encoded_point, direction, neighbour_point_plus, neighbour_point_minus, plus_overflow, minus_overflow);
    //     std::cout<<"direction: "<<direction<<", neighbour_point_plus: "<<neighbour_point_plus<<", neighbour_point_minus: "<<neighbour_point_minus<<", plus_overflow: "<<plus_overflow<<", minus_overflow: "<<minus_overflow<<"\n";
    // }
}
