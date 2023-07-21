add_executable(test_splat test_splat.cu)
target_link_libraries(test_splat attention_pert_lattice) 

add_executable(test_insert_and_find test_insert_and_find.cu)
target_link_libraries(test_insert_and_find cuco) 
target_compile_options(test_insert_and_find PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>)

add_executable(test_hashtable_lattice test_hashtable_lattice.cu)
target_link_libraries(test_hashtable_lattice cuco hashtable_lattice) 
target_compile_options(test_hashtable_lattice PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>)

add_executable(test_lattice_point_encoding test_lattice_point_encoding.cu)

add_executable(test_rank_rem0_barycentric test_rank_rem0_barycentric.cu)
target_link_libraries(test_rank_rem0_barycentric attention_pert_lattice) 

add_executable(test_cuco_custom_key test_cuco_custom_key.cu)
target_link_libraries(test_cuco_custom_key cuco) 
target_compile_options(test_cuco_custom_key PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>)
