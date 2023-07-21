#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cuco/static_map.cuh>
#include <thrust/sequence.h>
#include <thrust/count.h>

template<typename INT_TYPE>
struct key_with_dummy {
    INT_TYPE a = 0;
    INT_TYPE dummy_memory_occupier;
    __host__ __device__ key_with_dummy() {}
    __host__ __device__ key_with_dummy(INT_TYPE x) : a{x} {}
    __host__ __device__ bool operator==(key_with_dummy const& other) const { return a == other.a; }
};

template<typename INT_TYPE>
struct key_no_dummy {
    INT_TYPE a = 0;
    __host__ __device__ key_no_dummy() {}
    __host__ __device__ key_no_dummy(INT_TYPE x) : a{x} {}
    __host__ __device__ bool operator==(key_no_dummy const& other) const { return a == other.a; }
};

template<typename KEY_TYPE>
struct hasher_32 {
    __device__ uint32_t operator()(KEY_TYPE key) {
        cuco::murmurhash3_fmix_32<uint32_t> hash;
        return hash(key.a);
    };
};

template<typename KEY_TYPE>
struct hasher_64 {
    __device__ uint64_t operator()(KEY_TYPE key) {
        cuco::murmurhash3_fmix_64<uint64_t> hash;
        return hash(key.a);
    };
};

template<typename KEY_TYPE, typename HASHER>
struct populate_hash_table_device {
    typename cuco::static_map<KEY_TYPE, uint32_t>::device_mutable_view hasht_view;
    __device__ bool operator()(const uint32_t i) {
        return hasht_view.template insert<HASHER>(cuco::pair(i * 2000 + 1, i + 1));
    }
};

template<typename KEY_TYPE, typename HASHER>
const auto test_pair(const int num_entries, const int capacity)
{
    cuco::empty_key<KEY_TYPE> empty_key_sentinel{ KEY_TYPE(0) };
    cuco::empty_value<uint32_t> empty_value_sentinel{ 0 };
  
    cuco::static_map<KEY_TYPE, uint32_t> hash_table(capacity, empty_key_sentinel, empty_value_sentinel);

    populate_hash_table_device<KEY_TYPE, HASHER> populate_hashtable_func{hash_table.get_device_mutable_view()};

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const auto hashtable_size = thrust::count_if(
        thrust::make_counting_iterator<uint32_t>(0), 
        thrust::make_counting_iterator<uint32_t>(0) + num_entries, 
        populate_hashtable_func);

    thrust::device_vector<KEY_TYPE> retrieved_keys(hashtable_size);
    thrust::device_vector<uint32_t> retrieved_values(hashtable_size);
    auto [key_end, value_end] = hash_table.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());
    cudaDeviceSynchronize();
    const auto duration = std::chrono::steady_clock::now() - begin;
    return duration;
}

int main(int argc, char** argv)
{
    const int num_entries = 1000000;
    const int capacity = num_entries * 3;
    {
        test_pair<key_no_dummy<uint32_t>, hasher_32<key_no_dummy<uint32_t>>>(num_entries, capacity);
        cudaDeviceSynchronize();
        std::cout << "warm start\n";
    }
    {
        const auto duration = test_pair<key_no_dummy<uint32_t>, hasher_32<key_no_dummy<uint32_t>>>(num_entries, capacity);
        std::cout << "execution time for insert on UINT32 without dummy member is "<<std::chrono::duration_cast<std::chrono::microseconds>(duration).count()<<" us\n";
    }
    {
        const auto duration = test_pair<key_with_dummy<uint32_t>, hasher_32<key_with_dummy<uint32_t>>>(num_entries, capacity);
        std::cout << "execution time for insert on UINT32 with dummy member is "<<std::chrono::duration_cast<std::chrono::microseconds>(duration).count()<<" us\n";
    }
    {
        const auto duration = test_pair<key_no_dummy<uint64_t>, hasher_64<key_no_dummy<uint64_t>>>(num_entries, capacity);
        std::cout << "execution time for insert on UINT64 without dummy member is "<<std::chrono::duration_cast<std::chrono::microseconds>(duration).count()<<" us\n";
    }
    {
        const auto duration = test_pair<key_with_dummy<uint64_t>, hasher_64<key_with_dummy<uint64_t>>>(num_entries, capacity);
        std::cout << "execution time for insert on UINT64 with dummy member is "<<std::chrono::duration_cast<std::chrono::microseconds>(duration).count()<<" us\n";
    }
}
