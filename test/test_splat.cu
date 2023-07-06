#include <vector>
#include <thrust/device_vector.h>
#include "attention_pert_lattice.h"

int main(int argc, char** argv)
{

    const size_t num_points = 3;
    const size_t num_neighbours = 3;
    const size_t d_value = 2;
    const size_t num_splatted_points = 7;

    thrust::device_vector<float> values(num_points * d_value);
    thrust::device_vector<int> out_indices(num_points * num_neighbours);
    thrust::device_vector<float> barycentric(num_points * num_neighbours);
    thrust::device_vector<float> splatted_output(num_splatted_points * d_value);

    thrust::device_ptr<float> values_th = values.data();    
    thrust::device_ptr<int> out_indices_th = out_indices.data();    
    thrust::device_ptr<float> barycentric_th = barycentric.data();    
    thrust::device_ptr<float> splatted_output_th = splatted_output.data();

    splat_lattice(num_points, num_neighbours, d_value, num_splatted_points, values_th, out_indices_th, barycentric_th, splatted_output_th);
}
