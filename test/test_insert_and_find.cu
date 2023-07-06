#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

static constexpr int Iters = 10'000;

template <typename View>
__global__ void parallel_sum(View v)
{
  for (int i = 0; i < Iters; i++) {
    {
      auto [iter, inserted] = v.insert_and_find(thrust::make_pair(i, 1));
      // for debugging...
      // if (iter->second < 0) {
      //   asm("trap;");
      // }
      if (!inserted) { iter->second += 1; }
    }
  }
}

struct parallel_sum_th {
    cuco::static_map<int64_t, int64_t>::device_mutable_view v;
    __device__ void operator()(const size_t d)
    {
      for (int i = 0; i < Iters; i++) {
        {
          auto [iter, inserted] = v.insert_and_find(thrust::make_pair(i, 1));
          // for debugging...
          // if (iter->second < 0) {
          //   asm("trap;");
          // }
          if (!inserted) { iter->second += 1; }
        }
      }
    }
};

int main(int argc, char** argv)
{
  cuco::empty_key<int64_t> empty_key_sentinel{-1};
  cuco::empty_value<int64_t> empty_value_sentinel{-1};
  cuco::static_map<int64_t, int64_t> m(10 * Iters, empty_key_sentinel, empty_value_sentinel);

  static constexpr int Blocks  = 1024;
  static constexpr int Threads = 128;
  parallel_sum<<<Blocks, Threads>>>(m.get_device_mutable_view());
  // parallel_sum parallel_sum_f({m.get_device_mutable_view()});
  // const size_t num_workers = Blocks * Threads;
  // thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, parallel_sum_f);
  cudaDeviceSynchronize();

  std::cout<<"Populated hash table. Size: "<<m.get_size()<<", capacity: "<<m.get_capacity()<<"\n";

  thrust::device_vector<int64_t> d_keys(Iters);
  thrust::device_vector<int64_t> d_values(Iters);

  thrust::sequence(thrust::device, d_keys.begin(), d_keys.end());
  m.find(d_keys.begin(), d_keys.end(), d_values.begin());

  // cuco::test::all_of(d_values.begin(), d_values.end(), [] __device__(int64_t v) { return v == Blocks * Threads; });
}