#include "attention_pert_lattice.h"
#include "hashtable_lattice.h"
#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;

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
        long splatted_values_ptr, long blur_n1_ptr, long blur_n2_ptr)
{
    thrust::device_ptr<REAL> splatted_values_th = thrust::device_pointer_cast(reinterpret_cast<REAL*>(splatted_values_ptr));    
    thrust::device_ptr<int> blur_n1_th = thrust::device_pointer_cast(reinterpret_cast<int*>(blur_n1_ptr));    
    thrust::device_ptr<int> blur_n2_th = thrust::device_pointer_cast(reinterpret_cast<int*>(blur_n2_ptr));

    blur_lattice(num_lattice_points, num_directions, d_value, invalid_blur_neighbour_value, filter_coeff_self, filter_coeff_n1,
        splatted_values_th, blur_n1_th, blur_n2_th);
}

PYBIND11_MODULE(attention_pert_lattice_py, m) {
    m.doc() = "Bindings for attention by permutohedral filtering";
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
        long splatted_values_ptr, long blur_n1_ptr, long blur_n2_ptr)
        {
            blur<float>(num_lattice_points, num_directions, d_value, invalid_blur_neighbour_value, filter_coeff_self, filter_coeff_n1,
                splatted_values_ptr, blur_n1_ptr, blur_n2_ptr);
        });
    py::class_<hashtable_lattice>(m, "hashtable_lattice_py")
        .def(py::init([](
            const int batch_size, const int num_positions, const int d_pos, long rem0_ptr, long ranks_ptr) 
        {
            thrust::device_ptr<int> rem0_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(rem0_ptr));    
            thrust::device_ptr<int> ranks_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(ranks_ptr));    
            return hashtable_lattice(batch_size, num_positions, d_pos, rem0_dev_ptr, ranks_dev_ptr);
        }))
        .def("get_hashtable_size", &hashtable_lattice::get_hashtable_size)
        .def("make_values_contiguous", &hashtable_lattice::make_values_contiguous)
        .def("compute_splat_indices", [](
            const hashtable_lattice& htl, long rem0_ptr, long ranks_ptr, long splat_indices_ptr) 
        {
            thrust::device_ptr<int> rem0_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(rem0_ptr));    
            thrust::device_ptr<int> ranks_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(ranks_ptr));    
            thrust::device_ptr<VALUE_TYPE> splat_indices_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(splat_indices_ptr));    
            htl.get_splatting_indices(rem0_dev_ptr, ranks_dev_ptr, splat_indices_dev_ptr);
        })
        .def("compute_blur_neighbours", [](
            const hashtable_lattice& htl, long blur_n1_ptr, long blur_n2_ptr) 
        {
            thrust::device_ptr<VALUE_TYPE> blur_n1_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(blur_n1_ptr));    
            thrust::device_ptr<VALUE_TYPE> blur_n2_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<VALUE_TYPE*>(blur_n2_ptr));    
            htl.compute_blur_neighbours(blur_n1_dev_ptr, blur_n2_dev_ptr);
        })
        .def("test_lattice_point_encoder", [](
        const hashtable_lattice& htl, long rem0_ptr, long ranks_ptr) 
        {
            thrust::device_ptr<int> rem0_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(rem0_ptr));    
            thrust::device_ptr<int> ranks_dev_ptr = thrust::device_pointer_cast(reinterpret_cast<int*>(ranks_ptr));    
            return htl.test_lattice_point_encoder(rem0_dev_ptr, ranks_dev_ptr);
        });
}

