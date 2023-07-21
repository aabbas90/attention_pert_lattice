#include <vector>
#include <thrust/device_vector.h>
#include <map>
#include <numeric>
#include <algorithm>
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

void check_blur_neighbours(const thrust::device_vector<int>& lattice_points, const thrust::device_vector<int>& blur_nd, const int d_lattice, const bool is_positive)
{
    auto print_vector = [](const std::vector<int>& vec, const char* name) {
        std::cout<<name<<": ";
        for (const auto& element : vec) {
            std::cout << std::setw(2) << element << " ";
        }
        std::cout << std::endl;
    };

    auto print_matrix = [] (const std::vector<int>& v, const char* name, const int num_cols) {
        std::cout<<name<<":\n";
        const int num_rows = v.size() / num_cols;
        auto start_location = v.begin();
        for (int r = 0; r != num_rows; r++)
        {
            std::vector<int> row(start_location, start_location + num_cols);
            for (auto val : row)
                std::cout << std::setw(2) << val << " ";
            // thrust::copy(start_location, start_location + num_cols, std::ostream_iterator<T>(std::cout, " "));

            start_location += num_cols;
            std::cout<<"\n";
        }
    };
    auto get_point = [=](const int pt_index) -> std::vector<int>
    {
        std::vector<int> pt_wo_last_coordinate(lattice_points.begin() + pt_index * d_lattice, 
                                lattice_points.begin() + (pt_index + 1) * d_lattice);
        const int sum = std::accumulate(pt_wo_last_coordinate.begin() + 1, pt_wo_last_coordinate.end(), 0);
        pt_wo_last_coordinate.push_back(-sum); // insert last coordinate.
        return pt_wo_last_coordinate;
    };

    auto check_neighbour = [=](const std::vector<int> i, std::vector<int> j) -> std::tuple<bool, int>
    {
        std::vector<int> diff(i.size());
        std::transform(i.begin(), i.end(), j.begin(), diff.begin(), std::minus<int>());
        if (diff[0] != 0)
            return {false, 0};
        int direction = diff.size();
        const int required_offset_1 = is_positive ? -1: 1;
        const int required_offset_d = is_positive ? d_lattice - 1: -d_lattice + 1;
        for (int d = 1; d != diff.size(); ++d)
        {
            if (diff[d] != required_offset_1 && diff[d] != required_offset_d)
                return {false, d};
            else if (diff[d] == required_offset_d)
                direction = d;
        }
        return {true, direction - 1};
    };

    const int num_pts = lattice_points.size() / d_lattice;
    std::vector<int> expected_blur_neighbours(d_lattice * num_pts, num_pts);
    for (int i = 0; i != num_pts; ++i)
    {
        std::vector<int> pt_i = get_point(i);
        for (int j = 0; j != num_pts; ++j)
        {
            if (i == j)
                continue;
            std::vector<int> pt_j = get_point(j);
            bool is_neighbour;
            int index;
            std::tie(is_neighbour, index) = check_neighbour(pt_i, pt_j);
            if(is_neighbour)
            {
                const int output_index = index * num_pts + i;
                expected_blur_neighbours[output_index] = j;
                // std::cout<<"Expected neighbours: "<<i<<" "<<j<<"\n";
                // print_vector(pt_i, "\t pt_i");
                // print_vector(pt_j, "\t pt_j");
            }
        }
    }
    std::vector<int> blur_h(blur_nd.begin(), blur_nd.end());
    if (!std::equal(blur_h.begin(), blur_h.end(), expected_blur_neighbours.begin()))
    {
        print_matrix(expected_blur_neighbours, "expected_blur_neighbours", num_pts);
        print_matrix(blur_h, "computed_blur_neighbours", num_pts);
        std::cout<<"\n";
    }
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

    thrust::device_vector<int> min_coordinate_per_pos, cumulative_num_bits_per_dim;
    std::tie(min_coordinate_per_pos, cumulative_num_bits_per_dim) = calculate_lattice_extents(batch_size, num_positions, d_pos, rem0.data(), ranks.data());
    print_vector(min_coordinate_per_pos, "min_coordinate_per_pos");
    print_vector(cumulative_num_bits_per_dim, "cumulative_num_bits_per_dim");
    hashtable_lattice<1> htl(batch_size, num_positions, d_pos, min_coordinate_per_pos, cumulative_num_bits_per_dim);
    htl.add_points_to_lattice(rem0.data(), ranks.data());
    htl.make_values_contiguous();
    {
        const auto out = htl.get_valid_lattice_points_and_indices();
        thrust::device_vector<int> lattice_points = std::get<0>(out);
        std::vector<int> lattice_points_computed_h(lattice_points.begin(), lattice_points.end());
        thrust::device_vector<encoded_lattice_pt<1>> encoded_pts = std::get<1>(out);
        thrust::device_vector<int> indices = std::get<2>(out);
        print_matrix(lattice_points, "lattice_points", d_pos + 1);
        print_matrix(encoded_pts, "encoded_pts", 1);
        print_vector(indices, "indices");
        std::cout<<"Checking which computed key is not in reference.\n";
        check_rows(lattice_points_h, lattice_points_computed_h, d_pos + 1);
        std::cout<<"Checking which reference key is not in computed.\n";
        check_rows(lattice_points_computed_h, lattice_points_h, d_pos + 1);

        // std::vector<int> lattice_pts_exhaustive = htl.compute_all_lattice_points_slow(rem0.data(), ranks.data());    
        // check_rows(lattice_points_h, lattice_pts_exhaustive, d_pos + 1);

        thrust::device_vector<VALUE_TYPE> splatting_indices(batch_size * num_positions * (d_pos + 1));
        htl.get_splatting_indices(rem0.data(), ranks.data(), splatting_indices.data());
        thrust::device_vector<VALUE_TYPE> blur_n1((d_pos + 1) * htl.get_num_lattice_points());
        thrust::device_vector<VALUE_TYPE> blur_n2((d_pos + 1) * htl.get_num_lattice_points());
        htl.compute_blur_neighbours(blur_n1.data(), blur_n2.data());
        // htl.compute_blur_neighbours_direct(rem0.data(), ranks.data(), splatting_indices.data(), blur_n1.data(), blur_n2.data());
        // print_matrix(splatting_indices, "splatting_indices", d_pos + 1);
        // print_matrix(blur_n1, "blur_n1", htl.get_num_lattice_points());
        // print_matrix(blur_n2, "blur_n2", htl.get_num_lattice_points());
        thrust::device_vector<encoded_lattice_pt<1>> keys;
        thrust::device_vector<VALUE_TYPE> values;
        std::tie(keys, values) = htl.get_hashtable_entries();
        std::vector<int> cumulative_num_bits = htl.get_cumulative_num_bits();
        std::vector<int> min_coordinate = htl.get_min_coordinate_per_pos();
        check_blur_neighbours(lattice_points, blur_n2, d_pos + 1, true);
        check_blur_neighbours(lattice_points, blur_n1, d_pos + 1, false);
    }
    {
        thrust::device_vector<VALUE_TYPE> splatting_indices_direct(batch_size * num_positions * (d_pos + 1));
        hashtable_lattice<1> htl(batch_size, num_positions, d_pos, min_coordinate_per_pos, cumulative_num_bits_per_dim);
        htl.get_splatting_indices_direct(rem0.data(), ranks.data(), splatting_indices_direct.data());
        print_matrix(splatting_indices_direct, "splatting_indices_direct", d_pos + 1);
        // std::vector<int> lattice_pts_exhaustive = htl.compute_all_lattice_points_slow(rem0.data(), ranks.data());
        // auto print_std_vector = [](const std::vector<int>& vec, const char* name) {
        //     std::cout<<name<<": ";
        //     for (const auto& element : vec) 
        //     {
        //         std::cout << element << " ";
        //     }
        //     std::cout << std::endl;
        // };
        // print_std_vector(lattice_pts_exhaustive, "lattice_pts_exhaustive");

    //     // check_rows(lattice_points_h, lattice_pts_exhaustive, d_pos + 1);

    }
    // const std::vector<int> cumulative_num_bits_host = htl.get_cumulative_num_bits();

    // encoded_lattice_pt<1> neighbour_point_plus, neighbour_point_minus;
    // bool plus_overflow, minus_overflow;
    // const encoded_lattice_pt<1> encoded_point = 69;
    // for (int direction = 2; direction != 3; direction++)
    // {
    //     compute_neighbour_encoding(cumulative_num_bits_host.data(), d_pos + 1, encoded_point, direction, neighbour_point_plus, neighbour_point_minus, plus_overflow, minus_overflow);
    //     std::cout<<"direction: "<<direction<<", neighbour_point_plus: "<<neighbour_point_plus<<", neighbour_point_minus: "<<neighbour_point_minus<<", plus_overflow: "<<plus_overflow<<", minus_overflow: "<<minus_overflow<<"\n";
    // }
}
