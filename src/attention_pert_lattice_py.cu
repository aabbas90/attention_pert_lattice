#include "attention_pert_lattice.h"
#include "hashtable_wrapper.h"
#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;

template<typename REAL>
void compute_rem0_rank_barycentric(
        const size_t num_positions, const size_t d_lattice,
        const long features_ptr, const long out_rem0, const long out_ranks, const long out_barycentric)
{
    thrust::device_ptr<REAL> features_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(features_ptr));    
    thrust::device_ptr<int> rem0_th = thrust::device_pointer_cast(reinterpret_cast<int*>(out_rem0));
    thrust::device_ptr<int> ranks_th = thrust::device_pointer_cast(reinterpret_cast<int*>(out_ranks));
    thrust::device_ptr<REAL> barycentric_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(out_barycentric));
    compute_rem0_rank_barycentric(num_positions, d_lattice, features_th, rem0_th, ranks_th, barycentric_th);
}

template<typename REAL>
void splat(const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const long values_ptr, const long out_indices_ptr, const long barycentric_ptr, const long splatted_output_ptr)
{
    thrust::device_ptr<REAL> values_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(values_ptr));    
    thrust::device_ptr<int> out_indices_th = thrust::device_pointer_cast(reinterpret_cast<int*>(out_indices_ptr));    
    thrust::device_ptr<REAL> barycentric_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(barycentric_ptr));    
    thrust::device_ptr<REAL> splatted_output_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(splatted_output_ptr));

    splat_lattice(num_points, num_neighbours, d_value, num_splatted_points, values_th, out_indices_th, barycentric_th, splatted_output_th);
}

template<typename REAL>
void blur(const size_t num_lattice_points, const size_t num_directions, const size_t d_value, 
        const int invalid_blur_neighbour_value,
        const REAL filter_coeff_self, const REAL filter_coeff_n1,
        long splatted_values_ptr, long blur_n1_ptr, long blur_n2_ptr, const bool do_reverse)
{
    thrust::device_ptr<REAL> splatted_values_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(splatted_values_ptr));    
    thrust::device_ptr<int> blur_n1_th = thrust::device_pointer_cast(reinterpret_cast<int*>(blur_n1_ptr));    
    thrust::device_ptr<int> blur_n2_th = thrust::device_pointer_cast(reinterpret_cast<int*>(blur_n2_ptr));

    blur_lattice(num_lattice_points, num_directions, d_value, invalid_blur_neighbour_value, filter_coeff_self, filter_coeff_n1,
        splatted_values_th, blur_n1_th, blur_n2_th, do_reverse);
}

template<typename REAL>
void slice(const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const long splatted_input_ptr, const long in_indices_ptr, const long barycentric_ptr, const long sliced_values_ptr)
{
    thrust::device_ptr<REAL> splatted_input_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(splatted_input_ptr));
    thrust::device_ptr<int> in_indices_th = thrust::device_pointer_cast(reinterpret_cast<int*>(in_indices_ptr));    
    thrust::device_ptr<REAL> barycentric_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(barycentric_ptr));    
    thrust::device_ptr<REAL> sliced_values_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(sliced_values_ptr));    

    slice_lattice(num_points, num_neighbours, d_value, num_splatted_points, splatted_input_th, in_indices_th, barycentric_th, sliced_values_th);
}

template<typename REAL>
void grad_barycentric(const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const long splatted_input_ptr, const long in_indices_ptr, const long values_ptr, const long sliced_values_ptr)
{
    thrust::device_ptr<REAL> splatted_input_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(splatted_input_ptr));
    thrust::device_ptr<int> in_indices_th = thrust::device_pointer_cast(reinterpret_cast<int*>(in_indices_ptr));    
    thrust::device_ptr<REAL> values_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(values_ptr));    
    thrust::device_ptr<REAL> sliced_values_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(sliced_values_ptr));    

    grad_barycentric(num_points, num_neighbours, d_value, num_splatted_points, splatted_input_th, in_indices_th, values_th, sliced_values_th);
}

PYBIND11_MODULE(attention_pert_lattice_py, m) {
    m.doc() = "Bindings for attention by permutohedral filtering";
    m.def("compute_rem0_rank_barycentric", [](
        const size_t num_positions, const size_t d_lattice,
        const long features_ptr, const long out_rem0, const long out_ranks, const long out_barycentric)
        {
            compute_rem0_rank_barycentric<float>(num_positions, d_lattice,
                features_ptr, out_rem0, out_ranks, out_barycentric);
        });
    m.def("splat", [](
        const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const long values_ptr, const long out_indices_ptr, const long barycentric_ptr, const long splatted_output_ptr)
        {
            splat<float>(num_points, num_neighbours, d_value, num_splatted_points, values_ptr, out_indices_ptr, barycentric_ptr, splatted_output_ptr);
        });
    m.def("blur", [](
        const size_t num_lattice_points, const size_t num_directions, const size_t d_value, 
        const int invalid_blur_neighbour_value,
        const float filter_coeff_self, const float filter_coeff_n1,
        long splatted_values_ptr, long blur_n1_ptr, long blur_n2_ptr, const bool do_reverse)
        {
            blur<float>(num_lattice_points, num_directions, d_value, invalid_blur_neighbour_value, filter_coeff_self, filter_coeff_n1,
                splatted_values_ptr, blur_n1_ptr, blur_n2_ptr, do_reverse);
        });
    m.def("slice", [](
        const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const long splatted_input_ptr, const long in_indices_ptr, const long barycentric_ptr, const long sliced_output_ptr)
        {
            slice<float>(num_points, num_neighbours, d_value, num_splatted_points, splatted_input_ptr, in_indices_ptr, barycentric_ptr, sliced_output_ptr);
        });
    m.def("grad_barycentric", [](
        const size_t num_points, const size_t num_neighbours, const size_t d_value, const size_t num_splatted_points,
        const long splatted_input_ptr, const long in_indices_ptr, const long values_ptr, const long sliced_output_ptr)
        {
            grad_barycentric<float>(num_points, num_neighbours, d_value, num_splatted_points, splatted_input_ptr, in_indices_ptr, values_ptr, sliced_output_ptr);
        });

    py::class_<hashtable_wrapper>(m, "hashtable_lattice_py")
        .def(py::init([](
            const int batch_size, const int num_positions, const int d_pos, long rem0_ptr, long ranks_ptr) 
        {
            thrust::device_ptr<int> rem0_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(rem0_ptr));    
            thrust::device_ptr<int> ranks_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(ranks_ptr));    
            thrust::device_vector<int> min_coordinate_per_pos, cumulative_num_bits_per_dim;
            std::tie(min_coordinate_per_pos, cumulative_num_bits_per_dim) = calculate_lattice_extents(
                batch_size, num_positions, d_pos, rem0_dev_ptr, ranks_dev_ptr);
            return hashtable_wrapper(batch_size, num_positions, d_pos, min_coordinate_per_pos, cumulative_num_bits_per_dim);
        }))
        .def("add_points_to_lattice", [](
            hashtable_wrapper& htl, long rem0_ptr, long ranks_ptr) 
        {
            thrust::device_ptr<int> rem0_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(rem0_ptr));    
            thrust::device_ptr<int> ranks_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(ranks_ptr));    
            htl.add_points_to_lattice(rem0_dev_ptr, ranks_dev_ptr);
        })
        .def("get_num_lattice_points", &hashtable_wrapper::get_num_lattice_points)
        .def("make_values_contiguous", &hashtable_wrapper::make_values_contiguous)
        .def("compute_splat_indices", [](
            const hashtable_wrapper& htl, long rem0_ptr, long ranks_ptr, long splat_indices_ptr) 
        {
            thrust::device_ptr<int> rem0_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(rem0_ptr));    
            thrust::device_ptr<int> ranks_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(ranks_ptr));    
            thrust::device_ptr<VALUE_TYPE> splat_indices_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(splat_indices_ptr));    
            htl.get_splatting_indices(rem0_dev_ptr, ranks_dev_ptr, splat_indices_dev_ptr);
        })
        .def("compute_splat_indices_direct", [](
        hashtable_wrapper& htl, long rem0_ptr, long ranks_ptr, long splat_indices_ptr) 
        {
            thrust::device_ptr<int> rem0_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(rem0_ptr));    
            thrust::device_ptr<int> ranks_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(ranks_ptr));    
            thrust::device_ptr<VALUE_TYPE> splat_indices_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(splat_indices_ptr));    
            return htl.get_splatting_indices_direct(rem0_dev_ptr, ranks_dev_ptr, splat_indices_dev_ptr);
        })
        .def("compute_blur_neighbours", [](
            const hashtable_wrapper& htl, long blur_n1_ptr, long blur_n2_ptr) 
        {
            thrust::device_ptr<VALUE_TYPE> blur_n1_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(blur_n1_ptr));    
            thrust::device_ptr<VALUE_TYPE> blur_n2_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(blur_n2_ptr));    
            htl.compute_blur_neighbours(blur_n1_dev_ptr, blur_n2_dev_ptr);
        })
        .def("compute_blur_neighbours_direct", [](
        const hashtable_wrapper& htl, long rem0_ptr, long ranks_ptr, long splat_indices_ptr, long blur_n1_ptr, long blur_n2_ptr) 
        {
            thrust::device_ptr<int> rem0_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(rem0_ptr));    
            thrust::device_ptr<int> ranks_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(ranks_ptr));    
            thrust::device_ptr<VALUE_TYPE> splat_indices_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(splat_indices_ptr));    
            thrust::device_ptr<VALUE_TYPE> blur_n1_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(blur_n1_ptr));    
            thrust::device_ptr<VALUE_TYPE> blur_n2_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(blur_n2_ptr));    
            return htl.compute_blur_neighbours_direct(rem0_dev_ptr, ranks_dev_ptr, splat_indices_dev_ptr, blur_n1_dev_ptr, blur_n2_dev_ptr);
        })
        // .def("test_lattice_point_encoder", [](
        // const hashtable_lattice& htl, long rem0_ptr, long ranks_ptr) 
        // {
        //     thrust::device_ptr<int> rem0_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(rem0_ptr));    
        //     thrust::device_ptr<int> ranks_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(ranks_ptr));    
        //     return htl.test_lattice_point_encoder(rem0_dev_ptr, ranks_dev_ptr);
        // })
        ;
}

