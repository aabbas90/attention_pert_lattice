add_library(hashtable_lattice hashtable_lattice.cu)
target_link_libraries(hashtable_lattice PUBLIC cuco) 
target_compile_options(hashtable_lattice PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>)

add_library(attention_pert_lattice attention_pert_lattice.cu)
target_compile_options(attention_pert_lattice PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>)

pybind11_add_module(attention_pert_lattice_py attention_pert_lattice_py.cu)
target_link_libraries(attention_pert_lattice_py PRIVATE attention_pert_lattice hashtable_lattice) 