#include <vector>
#include <map>
#include "cuda_utils.h"

template<typename T>
T encode_point(
    const int* const cumulative_num_bits, const int* const min_c, const int* const coordinates, 
    const int batch_index, const T reminder, const int d_lattice)
{
    T packedNumber = 0;
    int c = 0;
    int shift = 0;
    for (c = 0; c != d_lattice - 1; c++)
    {
        int pt_coordinate = coordinates[c];
        if (pt_coordinate != 0)
            pt_coordinate = floor_divisor(pt_coordinate, d_lattice);
        pt_coordinate -= min_c[c];
        assert(pt_coordinate >= 0);
        // Pack the number by shifting it and combining with the packedNumber
        packedNumber |= pt_coordinate << shift;
        std::cout<<"c: "<<c<<", packedNumber: "<<packedNumber<<"\n";
        shift = cumulative_num_bits[c];
    }
    std::cout<<"reminder << shift: "<<reminder << shift<<"\n";
    packedNumber |= reminder << shift;
    packedNumber |= batch_index << cumulative_num_bits[d_lattice - 1];
    return packedNumber;
}


int main(int argc, char** argv)
{
    std::vector<int> cumulative_num_bits_per_dim = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 52, 53};
    std::vector<int> min_coordinate_per_pos = {-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2};
    std::vector<int> p1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -16, 1};
    std::vector<int> p2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, -17, 0};

    const auto p1_encoding = encode_point<uint64_t>(cumulative_num_bits_per_dim.data(), min_coordinate_per_pos.data(), p1.data(), 0, 1, 17);
    std::cout<<"p1_encoding: "<<p1_encoding<<"\n";

    const auto p2_encoding = encode_point<uint64_t>(cumulative_num_bits_per_dim.data(), min_coordinate_per_pos.data(), p2.data(), 0, 0, 17);
    std::cout<<"p2_encoding: "<<p2_encoding<<"\n";
}

