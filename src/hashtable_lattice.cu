#include "cuda_utils.h"
#include "time_measure_util.h"
#include "hashtable_lattice.h"
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/sort.h>

void update_permutation(const uint64_t* const keys, thrust::device_vector<uint32_t>& permutation)
{
    // temporary storage for keys
    thrust::device_vector<uint64_t> temp(permutation.size());

    // permute the keys with the current reordering
    thrust::device_ptr<const uint64_t> keys_ptr = thrust::device_pointer_cast(keys);
    thrust::gather(permutation.begin(), permutation.end(), keys_ptr, temp.begin());

    // stable_sort the permuted keys and update the permutation
    thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}

template <typename T>
void apply_permutation(T* keys, thrust::device_vector<uint32_t>& permutation)
{
    // copy keys to temporary vector
    thrust::device_vector<T> temp(keys, keys + permutation.size());

    // permute the keys
    thrust::device_ptr<T> keys_ptr = thrust::device_pointer_cast(keys);
    thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys_ptr);
}

struct compute_max_min_coordinates {
    const int total_num_p; // includes batch size.
    const int d_pos;
    const int* const rem0;
    const int* const ranks;
    int* minimum_coordinates;
    int* maximum_coordinates;
    __device__ void operator()(const size_t i)
    {
        // TODOAA: Profile by transposing following two:
        // const int index_pt = i % total_num_p;
        // const int index_d = i / total_num_p;
        const int index_d = i % d_pos;
        const int index_pt = i / d_pos;
        assert (index_d < d_pos); // although lattice has dim = d_pos + 1, but only d_pos needs to be computed.
        const int current_rank = ranks[index_pt * (d_pos + 1) + index_d];
        const int current_rem0 = rem0[index_pt * (d_pos + 1) + index_d];
        int current_min, current_max;
        for (int r = 0; r != d_pos + 1; r++)
        {
            int pt_coordinate = current_rem0 + compute_canonical_simplex_point_coordinate(r, current_rank, d_pos);
            if (pt_coordinate != 0)
                pt_coordinate = floor_divisor(pt_coordinate, d_pos + 1);

            if (r == 0) {
                current_min = pt_coordinate;
                current_max = pt_coordinate;
            } else {
                current_min = min(current_min, pt_coordinate);
                current_max = max(current_max, pt_coordinate);
            }
        }
        atomicMin(&minimum_coordinates[index_d], current_min);
        atomicMax(&maximum_coordinates[index_d], current_max);
    }
};

struct count_bits {
    const int* minimum_coordinates;
    const int* maximum_coordinates;
    int* num_bits;
    __host__ __device__ void operator()(const size_t i)
    {
        num_bits[i] = (int) log2(maximum_coordinates[i] - minimum_coordinates[i] + 1) + 1;
    }
};

// rem0: [batch_size, num_positions, d_pos + 1]
// ranks: [batch_size, num_positions, d_pos + 1]
std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> calculate_lattice_extents(
    const int batch_size, const int num_positions, const int d_pos, 
    const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks)
{
    thrust::device_vector<int> min_coordinate_per_pos(d_pos, std::numeric_limits<int>::max());
    thrust::device_vector<int> max_coordinate_per_pos(d_pos, std::numeric_limits<int>::min());

    const size_t num_workers = batch_size * num_positions * d_pos;
    compute_max_min_coordinates max_min_func({batch_size * num_positions, d_pos,
                        thrust::raw_pointer_cast(rem0), 
                        thrust::raw_pointer_cast(ranks), 
                        thrust::raw_pointer_cast(min_coordinate_per_pos.data()),
                        thrust::raw_pointer_cast(max_coordinate_per_pos.data())});

    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, max_min_func);

    const int d_encoded = d_pos + 2;
    thrust::device_vector<int> cumulative_num_bits_per_dim(d_encoded, 0);

    const int d_lattice = d_pos + 1;
    cumulative_num_bits_per_dim[d_encoded - 2] = (int) std::log2(d_lattice) + 1;
    cumulative_num_bits_per_dim[d_encoded - 1] = (int) std::log2(batch_size) + 1 + 1; // TODOAA: +1 for empty stencil

    // Compute number of bits required per each encoded dimension.
    count_bits count_bits_func({
                        thrust::raw_pointer_cast(min_coordinate_per_pos.data()),
                        thrust::raw_pointer_cast(max_coordinate_per_pos.data()),
                        thrust::raw_pointer_cast(cumulative_num_bits_per_dim.data())});

    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + d_pos, count_bits_func);

    // Convert to cumulative:
    thrust::inclusive_scan(cumulative_num_bits_per_dim.begin(), cumulative_num_bits_per_dim.end(), cumulative_num_bits_per_dim.begin());
    std::vector<int> cumulative_num_bits_per_dim_h(cumulative_num_bits_per_dim.begin(), cumulative_num_bits_per_dim.end());
    // print_vector(cumulative_num_bits_per_dim, "cumulative_num_bits_per_dim before");
    for (int i = 0; i < cumulative_num_bits_per_dim_h.size() - 1; i++)
    {
        const int pi = cumulative_num_bits_per_dim_h[i] / 64;
        const int pj = cumulative_num_bits_per_dim_h[i + 1] / 64;
        if (pi != pj)
        {
            const int offset = pj * 64 - cumulative_num_bits_per_dim_h[i];
            for (int j = i; j < cumulative_num_bits_per_dim_h.size(); j++)
                cumulative_num_bits_per_dim_h[j] += offset;
        }
    }
    thrust::copy(cumulative_num_bits_per_dim_h.begin(), cumulative_num_bits_per_dim_h.end(), cumulative_num_bits_per_dim.begin());
    return {min_coordinate_per_pos, cumulative_num_bits_per_dim};
}

template<int NUM_VECS> 
hashtable_lattice<NUM_VECS>::hashtable_lattice(
    const int _batch_size, const int _num_positions, const int _d_pos, 
    const thrust::device_vector<int>& _min_coordinate_per_pos, const thrust::device_vector<int>& _cumulative_num_bits_per_dim) :
    batch_size(_batch_size), num_positions(_num_positions), d_pos(_d_pos), d_encoded(_d_pos + 2),
    min_coordinate_per_pos(_min_coordinate_per_pos), cumulative_num_bits_per_dim(_cumulative_num_bits_per_dim)
{
    number_of_required_bits = cumulative_num_bits_per_dim.back();
    // d_encoded = 1 + 1 + d_pos, where +1 for batch index, another +1 for reminder, 
    // rest d_pos for lattice point. Since lattice points sum to 0 so using d_pos instead of d_pos + 1.
}

template <int N>
struct struct_of_arrays
{
    std::array<uint64_t*, N> data;
};

template <int N>
struct get_encoded_points_kernel {
    const int batch_size;
    const int num_positions;
    const int d_lattice;
    const int* const rem0;
    const int* const ranks;
    const int* const cumulative_num_bits;
    const int* const min_coordinate;
    // KEY_TYPE* encoded_points;
    struct_of_arrays<N> encoded_points;
    uint32_t* sorting_order;
    __device__ bool operator()(const size_t i)
    {
        const int reminder = i % d_lattice;
        const int index_pt = (i / d_lattice) % num_positions;
        const int batch_index = i / (d_lattice * num_positions);
        const int start_index = (batch_index * num_positions + index_pt) * d_lattice;
        const encoded_lattice_pt<N> encoded_pt = encode_point<encoded_lattice_pt<N>>(
            cumulative_num_bits, min_coordinate, rem0, ranks, batch_index, start_index, reminder, d_lattice);
        // writing encoded_points in the following order to ease sorting:
        // batch_index, reminder, point_index
        const size_t out_index = (batch_index * num_positions * d_lattice) + (reminder * num_positions) + index_pt;
        for (int i = 0; i != N; ++i)
            encoded_points.data[i][out_index] = encoded_pt.data[i];
        sorting_order[out_index] = i;
    }
};

template <int N>
struct mark_first_unique_lattice_pt {
    const struct_of_arrays<N> encoded_points;
    uint32_t* is_unique;
    __device__ bool operator()(const size_t i)
    {
        assert(i > 0);
        bool cur_is_unique = false;
        for (int s = 0; s != N; ++s)
        {
            if(encoded_points.data[s][i] != encoded_points.data[s][i - 1])
            {
                cur_is_unique = true;
                break;
            }
        }
        if (cur_is_unique)
            is_unique[i] = 1;
    }
};

// output should be preallocated to size batch_size * num_positions * (d_pos + 1).
template<int NUM_VECS>
int hashtable_lattice<NUM_VECS>::get_splatting_indices_direct(
    const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output)
{
    // Each input point splats to exactly d_pos + 1 many lattice points.
    const size_t num_workers = batch_size * num_positions * (d_pos + 1);

    struct_of_arrays<NUM_VECS> encoded_pts;

    for (int i = 0; i != NUM_VECS;  i++)
        cudaMalloc(&encoded_pts.data[i], num_workers * sizeof(uint64_t));

    thrust::device_vector<uint32_t> sorting_order(num_workers);
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("[get_splatting_indices_direct] get_encoded_points_kernel");
        get_encoded_points_kernel<NUM_VECS> get_encoded_points_func({batch_size, num_positions, d_pos + 1, 
            thrust::raw_pointer_cast(rem0),
            thrust::raw_pointer_cast(ranks),        
            thrust::raw_pointer_cast(cumulative_num_bits_per_dim.data()),
            thrust::raw_pointer_cast(min_coordinate_per_pos.data()),
            encoded_pts,
            thrust::raw_pointer_cast(sorting_order.data())});

        thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, get_encoded_points_func);
    }

    {
        std::string message = "[get_splatting_indices_direct] sort encoded points, encoding size: ";
        message += std::to_string(NUM_VECS) + " x 8bytes, ";
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2(message);

        // sort from least significant key to most significant keys
        thrust::device_vector<uint32_t> permutation(num_workers);
        thrust::sequence(permutation.begin(), permutation.end());
        for (int i = 0; i != NUM_VECS;  i++)
            update_permutation(encoded_pts.data[i], permutation);

        for (int i = 0; i != NUM_VECS;  i++)
            apply_permutation(encoded_pts.data[i], permutation);
        apply_permutation(thrust::raw_pointer_cast(sorting_order.data()), permutation);
    }

    thrust::device_vector<uint32_t> is_unique_pt(num_workers, 0);
    mark_first_unique_lattice_pt<NUM_VECS> mark_first_unique_lattice_pt_func({
        encoded_pts,
        thrust::raw_pointer_cast(is_unique_pt.data())
    });

    thrust::for_each(thrust::make_counting_iterator<size_t>(0) + 1, thrust::make_counting_iterator<size_t>(0) + num_workers,
                    mark_first_unique_lattice_pt_func);

    // convert to index:
    thrust::inclusive_scan(is_unique_pt.begin(), is_unique_pt.end(), is_unique_pt.begin());

    num_lattice_points = is_unique_pt.back() + 1;
    for (int i = 0; i != NUM_VECS;  i++)
        cudaFree(encoded_pts.data[i]);
    
    thrust::scatter(is_unique_pt.begin(), is_unique_pt.end(), sorting_order.begin(), output);
    return num_lattice_points;
}

template<int N>
struct compute_blur_neighbours_direct_kernel {
    const int num_lattice_points;
    const int d_lattice;
    const int* const rem0;
    const int* const ranks;
    const int* const splatting_table;
    VALUE_TYPE* const n1;
    VALUE_TYPE* const n2;
    __device__ void operator()(const size_t i)
    {
        const int self_reminder = i % d_lattice;
        const int start_index = i - self_reminder;
        const int self_splat_index = splatting_table[i];

        for(int other_reminder = self_reminder - 1; other_reminder <= self_reminder + 1; ++other_reminder)
        {
            if (other_reminder == self_reminder)
                continue;

            const int other_reminder_pos = positive_modulo(other_reminder, d_lattice);
            int blur_direction = 0;
            for (int index_d = 0; index_d != d_lattice; index_d++)
            {
                const int cur_rank = ranks[start_index + index_d];
                const int self_pt_coordinate = compute_canonical_simplex_point_coordinate(self_reminder, cur_rank, d_lattice - 1);
                const int other_pt_coordinate = compute_canonical_simplex_point_coordinate(other_reminder_pos, cur_rank, d_lattice - 1);
                const int diff = other_pt_coordinate - self_pt_coordinate;
                if (abs(diff) == 1)
                    continue;
                blur_direction = index_d;
                break;
            }
            const int other_splat_index = splatting_table[start_index + other_reminder_pos];
            const VALUE_TYPE output_index = blur_direction * num_lattice_points + self_splat_index;
            // if (i < 1000)
                // printf("self_pt_coordinate: %d, other_pt_coordinate: %d, diff: %d, self_splat_index: %d, other_splat_index: %d, self_reminder: %d, other_reminder: %d, other_reminder_pos: %d\n", 
                // self_pt_coordinate, other_pt_coordinate, diff, self_splat_index, other_splat_index, self_reminder, other_reminder, other_reminder_pos);
                // printf("self_splat_index: %d, other_splat_index: %d, self_reminder: %d, other_reminder_pos: %d, output_index, : %d\n", 
                //         self_splat_index, other_splat_index, self_reminder, other_reminder_pos, output_index);
            if (other_reminder < self_reminder)
                n1[output_index] = other_splat_index;
            else
                n2[output_index] = other_splat_index;
        }
    }
};

// rem0: [batch_size, num_positions, d_pos + 1]
// ranks: [batch_size, num_positions, d_pos + 1]
// splatting_table: [batch_size, num_positions, (d_pos + 1)]
// blur_n1: [d_lattice, num_splatted_points] should be preallocated.
// blur_n2: [d_lattice, num_splatted_points] should be preallocated.
template<int NUM_VECS>
void hashtable_lattice<NUM_VECS>::compute_blur_neighbours_direct(
    const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> splatting_table, 
    thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const
{
    const size_t num_workers = batch_size * num_positions * (d_pos + 1);
    thrust::fill(blur_n1, blur_n1 + (num_lattice_points * (d_pos + 1)), num_lattice_points); // values = num_lattice_points indicate no neighbour found.
    thrust::fill(blur_n2, blur_n2 + (num_lattice_points * (d_pos + 1)), num_lattice_points);

    compute_blur_neighbours_direct_kernel<NUM_VECS> compute_blur_neighbours_func({
        num_lattice_points, d_pos + 1, 
        thrust::raw_pointer_cast(rem0),
        thrust::raw_pointer_cast(ranks),
        thrust::raw_pointer_cast(splatting_table),
        thrust::raw_pointer_cast(blur_n1), 
        thrust::raw_pointer_cast(blur_n2)});

    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, compute_blur_neighbours_func);
}

template class hashtable_lattice<1>;
template class hashtable_lattice<2>;
template class hashtable_lattice<3>;
template class hashtable_lattice<4>;
template class hashtable_lattice<5>;
template class hashtable_lattice<6>;
template class hashtable_lattice<7>;
template class hashtable_lattice<8>;
template class hashtable_lattice<9>;