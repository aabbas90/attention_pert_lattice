#include "cuda_utils.h"
#include "time_measure_util.h"
#include "hashtable_lattice.h"
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <map>

// template<typename KEY_TYPE, typename VALUE_TYPE>
// hashtable_lattice<KEY_TYPE, VALUE_TYPE>
hashtable_lattice::hashtable_lattice(
    const int _batch_size, const int _num_positions, const int _d_pos, 
    const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks) :
    batch_size(_batch_size), num_positions(_num_positions), d_pos(_d_pos), d_encoded(_d_pos + 2)
{
    // d_encoded = 1 + 1 + d_pos, where +1 for batch index, another +1 for reminder, 
    // rest d_pos for lattice point. Since lattice points sum to 0 so using d_pos instead of d_pos + 1.
    
    // Calculate extents of the lattice, allowing to encode lattice points.
    populate_lattice_point_bits(rem0, ranks);
    add_points_to_lattice(rem0, ranks);
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
        const int index_pt = i % total_num_p;
        const int index_d = i / total_num_p;
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
// template<typename KEY_TYPE, typename VALUE_TYPE>
// void hashtable_lattice<KEY_TYPE, VALUE_TYPE>::
void hashtable_lattice::populate_lattice_point_bits(
    const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks)
{
    min_coordinate_per_pos = thrust::device_vector<int>(d_pos, std::numeric_limits<int>::max());
    max_coordinate_per_pos = thrust::device_vector<int>(d_pos, std::numeric_limits<int>::min());

    const size_t num_workers = batch_size * num_positions * d_pos;
    compute_max_min_coordinates max_min_func({batch_size * num_positions, d_pos,
                        thrust::raw_pointer_cast(rem0), 
                        thrust::raw_pointer_cast(ranks), 
                        thrust::raw_pointer_cast(min_coordinate_per_pos.data()),
                        thrust::raw_pointer_cast(max_coordinate_per_pos.data())});

    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, max_min_func);
    cumulative_num_bits_per_dim = thrust::device_vector<int>(d_encoded, 0);

    const int d_lattice = d_pos + 1;
    cumulative_num_bits_per_dim[d_encoded - 2] = (int) std::log2(d_lattice) + 1;
    cumulative_num_bits_per_dim[d_encoded - 1] = (int) std::log2(batch_size) + 1;

    // Compute number of bits required per each encoded dimension.
    count_bits count_bits_func({
                        thrust::raw_pointer_cast(min_coordinate_per_pos.data()),
                        thrust::raw_pointer_cast(max_coordinate_per_pos.data()),
                        thrust::raw_pointer_cast(cumulative_num_bits_per_dim.data())});

    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + d_pos, count_bits_func);

    int num_bits = thrust::reduce(cumulative_num_bits_per_dim.begin(), cumulative_num_bits_per_dim.end());
    const size_t encoding_capacity = sizeof(KEY_TYPE) * 8;

    std::cout<<"[lattice_point_converter] Lattice points representable in "<<num_bits<<" bits."<<"\n";
    if (num_bits > encoding_capacity)
    {
        throw std::runtime_error("Number of bits required: " + std::to_string(num_bits) + " > " + std::to_string(encoding_capacity));
    }
    // Convert to cumulative:
    thrust::inclusive_scan(cumulative_num_bits_per_dim.begin(), cumulative_num_bits_per_dim.end(), cumulative_num_bits_per_dim.begin());
}

// template<typename KEY_TYPE, typename VALUE_TYPE>
struct populate_hash_table {
    const int batch_size;
    const int num_positions;
    const int d_lattice;
    const int* const rem0;
    const int* const ranks;
    const int* const cumulative_num_bits;
    const int* const min_coordinate;
    cuco::static_map<KEY_TYPE, VALUE_TYPE>::device_mutable_view hasht_view;
    __device__ bool operator()(const size_t i)
    {
        const int reminder = i % d_lattice;
        const int index_pt = (i / d_lattice) % num_positions;
        const int batch_index = i / (d_lattice * num_positions);
        const int start_index = (batch_index * num_positions + index_pt) * d_lattice;
        KEY_TYPE encoded_pt = encode_point<KEY_TYPE>(
            cumulative_num_bits, min_coordinate, rem0, ranks, batch_index, start_index, reminder, d_lattice);
        auto [iter, inserted] = hasht_view.insert_and_find(thrust::make_pair(encoded_pt, (VALUE_TYPE) i));
        if (!inserted)
            iter->second = min(iter->second, (int) i);
        return inserted;
    }
};

//template<typename KEY_TYPE, typename VALUE_TYPE>
//void hashtable_lattice<KEY_TYPE, VALUE_TYPE>
void hashtable_lattice::add_points_to_lattice(const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks)
{
    // Create hashtable with maximum capacity as each point can splat to d_pos + 1 many points.
    hash_table = std::make_unique<cuco::static_map<KEY_TYPE, VALUE_TYPE>>(2 * batch_size * num_positions * (d_pos + 1), empty_key_sentinel, empty_value_sentinel);

    populate_hash_table populate_hashtable_func({
        batch_size, num_positions, d_pos + 1, 
        thrust::raw_pointer_cast(rem0),
        thrust::raw_pointer_cast(ranks),
        thrust::raw_pointer_cast(cumulative_num_bits_per_dim.data()),
        thrust::raw_pointer_cast(min_coordinate_per_pos.data()),
        hash_table->get_device_mutable_view()});
    
    const size_t num_workers = batch_size * num_positions * (d_pos + 1);
    hashtable_size = thrust::count_if(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, populate_hashtable_func);

    std::cout<<"Populated hash table. Size: "<<hashtable_size<<", capacity: "<<hash_table->get_capacity()<<"\n";
}

struct make_values_consecutive {
    const KEY_TYPE* const keys;
    const VALUE_TYPE* const values;
    cuco::static_map<KEY_TYPE, VALUE_TYPE>::device_view hasht_view;
    __device__ void operator()(const size_t i)
    {
        auto iter = hasht_view.find(keys[i]);
        iter->second = i;
    }
};

void hashtable_lattice::make_values_contiguous()
{
    // Now make values of hashtable lie in {0, 1, ..., hashtable_size - 1} thus allowing all subsequent operations on the lattice use less memory.
    // For that sort hash table values (with keys as keys) and then use index of this sorted array to compute new index. Afterwards call thrust struct to
    // update hash table values by doing find() on this sorted keys and replace by their corresponding values. 
    thrust::device_vector<KEY_TYPE> keys;
    thrust::device_vector<VALUE_TYPE> values;
    std::tie(keys, values) = this->get_hashtable_entries();

    thrust::sort_by_key(values.begin(), values.end(), keys.begin());
    make_values_consecutive make_values_consecutive_func({
        thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(values.data()),
        hash_table->get_device_view()});

    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + keys.size(), make_values_consecutive_func);
}

struct decode_points_from_keys {
    const int d_lattice;
    const int* const cumulative_num_bits;
    const int* const min_coordinate;
    const KEY_TYPE* const encoded_keys;
    int* decoded_lattice_points;
    __device__ void operator()(const size_t index_pt)
    {
        decode_point<KEY_TYPE>(cumulative_num_bits, min_coordinate, index_pt, d_lattice, encoded_keys[index_pt], decoded_lattice_points);
    }
};

std::tuple<thrust::device_vector<KEY_TYPE>, thrust::device_vector<VALUE_TYPE>> hashtable_lattice::get_hashtable_entries() const
{
    thrust::device_vector<KEY_TYPE> keys(hashtable_size);
    thrust::device_vector<VALUE_TYPE> values(hashtable_size);

    auto [key_end, value_end] = hash_table->retrieve_all(keys.begin(), values.begin());
    assert(std::distance(keys.begin(), key_end) == keys.size());
    assert(std::distance(values.begin(), value_end) == keys.size());
    return {keys, values};
}

std::tuple<thrust::device_vector<int>, thrust::device_vector<VALUE_TYPE>> hashtable_lattice::get_valid_lattice_points_and_indices() const
{
    thrust::device_vector<KEY_TYPE> keys;
    thrust::device_vector<VALUE_TYPE> values;
    std::tie(keys, values) = this->get_hashtable_entries();

    thrust::device_vector<int> lattice_points(hashtable_size * (d_pos + 1));
    decode_points_from_keys decode_points_from_keys_func({
        d_pos + 1, 
        thrust::raw_pointer_cast(cumulative_num_bits_per_dim.data()),
        thrust::raw_pointer_cast(min_coordinate_per_pos.data()),
        thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(lattice_points.data())});
    
    const size_t num_workers = hashtable_size;
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, decode_points_from_keys_func);
    return {lattice_points, values};
}

struct get_splatting_indices_kernel {
    const int batch_size;
    const int num_positions;
    const int d_lattice;
    const int* const rem0;
    const int* const ranks;
    const int* const cumulative_num_bits;
    const int* const min_coordinate;
    cuco::static_map<KEY_TYPE, VALUE_TYPE>::device_view hasht_view;
    VALUE_TYPE* splatting_indices;
    __device__ bool operator()(const size_t i)
    {
        const int reminder = i % d_lattice;
        const int index_pt = (i / d_lattice) % num_positions;
        const int batch_index = i / (d_lattice * num_positions);
        const int start_index = (batch_index * num_positions + index_pt) * d_lattice;
        KEY_TYPE encoded_pt = encode_point<KEY_TYPE>(
            cumulative_num_bits, min_coordinate, rem0, ranks, batch_index, start_index, reminder, d_lattice);
        const auto iter = hasht_view.find(encoded_pt);
        assert(iter != hasht_view.end()); // key should exist.
        splatting_indices[i] = iter->second;
    }
};

// output should be preallocated to size batch_size * num_positions * (d_pos + 1).
void hashtable_lattice::get_splatting_indices(
    const thrust::device_ptr<int> rem0, const thrust::device_ptr<int> ranks, thrust::device_ptr<VALUE_TYPE> output) const
{
    // Each input point splats to exactly d_pos + 1 many lattice points.
    const size_t num_workers = batch_size * num_positions * (d_pos + 1);
    get_splatting_indices_kernel get_splatting_indices_func({batch_size, num_positions, d_pos + 1, 
        thrust::raw_pointer_cast(rem0),
        thrust::raw_pointer_cast(ranks),        
        thrust::raw_pointer_cast(cumulative_num_bits_per_dim.data()),
        thrust::raw_pointer_cast(min_coordinate_per_pos.data()),
        hash_table->get_device_view(),
        thrust::raw_pointer_cast(output)});
    
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, get_splatting_indices_func);
}

struct compute_blur_neighbours_kernel {
    const int num_pts;
    const int d_lattice;
    const KEY_TYPE* encoded_keys;
    const VALUE_TYPE* values;
    const int* const cumulative_num_bits;
    cuco::static_map<KEY_TYPE, VALUE_TYPE>::device_view hasht_view;
    VALUE_TYPE* const n1;
    VALUE_TYPE* const n2;
    __device__ void operator()(const size_t i)
    {
        // // consecutive threads will operate on consecutive points and same direction. // TODOAA: benchmark.
        // const int index_pt = i % num_pts;
        // const int direction = i / num_pts;
        // consecutive threads will operate on same point with different direction.
        const int direction = i % d_lattice;
        const int index_pt = i / d_lattice;
        KEY_TYPE neighbour_point_plus, neighbour_point_minus;
        bool plus_overflow, minus_overflow;
        compute_neighbour_encoding<KEY_TYPE>(cumulative_num_bits, d_lattice, encoded_keys[index_pt], direction, 
            neighbour_point_plus, neighbour_point_minus, plus_overflow, minus_overflow);
        const VALUE_TYPE output_index = direction * num_pts + values[index_pt];
        if (!plus_overflow)
        {
            const auto iter = hasht_view.find(neighbour_point_plus);
            if(iter != hasht_view.end())
                n1[output_index] = iter->second;
        }
        if (!minus_overflow)
        {
            const auto iter = hasht_view.find(neighbour_point_minus);
            if(iter != hasht_view.end())
                n2[output_index] = iter->second;
        }
    }
};

//     blur_n1: [d_lattice, num_splatted_points] should be preallocated.
//     blur_n2: [d_lattice, num_splatted_points] should be preallocated.
void hashtable_lattice::compute_blur_neighbours(thrust::device_ptr<VALUE_TYPE> blur_n1, thrust::device_ptr<VALUE_TYPE> blur_n2) const
{
    size_t num_workers = (d_pos + 1) * hashtable_size;
    thrust::fill(blur_n1, blur_n1 + num_workers, hashtable_size); // values = hashtable_size indicate no neighbour found.
    thrust::fill(blur_n2, blur_n2 + num_workers, hashtable_size);
    thrust::device_vector<KEY_TYPE> keys;
    thrust::device_vector<VALUE_TYPE> values;
    std::tie(keys, values) = this->get_hashtable_entries();

    compute_blur_neighbours_kernel compute_blur_neighbours_func({hashtable_size, d_pos + 1, 
        thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(cumulative_num_bits_per_dim.data()),
        hash_table->get_device_view(),
        thrust::raw_pointer_cast(blur_n1), 
        thrust::raw_pointer_cast(blur_n2)});

    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(0) + num_workers, compute_blur_neighbours_func);
}


std::vector<int> hashtable_lattice::compute_all_lattice_points_slow(const thrust::device_ptr<int> rem0_d, const thrust::device_ptr<int> ranks_d) const
{
    std::vector<int> rem0(rem0_d, rem0_d + batch_size * num_positions * (d_pos + 1));
    std::vector<int> ranks(ranks_d, ranks_d + batch_size * num_positions * (d_pos + 1));
    std::vector<int> min_c(min_coordinate_per_pos.begin(), min_coordinate_per_pos.end());
    std::vector<int> cumulative_num_bits(cumulative_num_bits_per_dim.begin(), cumulative_num_bits_per_dim.end());
    std::vector<int> output;
    for(int index_pt = 0; index_pt != batch_size * num_positions; ++index_pt)
    {
        const int batch_index = index_pt / num_positions;
        for (int r = 0; r != d_pos + 1; r++)
        {
            std::cout<<"batch_index: "<<batch_index<<", index_pt: "<<index_pt<<", r: "<<r<<"\n\t";
            output.push_back(batch_index);
            uint32_t packedNumber = 0;
            int shift = 0;
            for(int index_d = 0; index_d != d_pos; ++index_d)
            {
                const int current_rank = ranks[index_pt * (d_pos + 1) + index_d];
                const int current_rem0 = rem0[index_pt * (d_pos + 1) + index_d];
                int coord = current_rem0 + compute_canonical_simplex_point_coordinate(r, current_rank, d_pos);
                std::cout<<" "<<coord;
                output.push_back(coord);

                if (coord != 0)
                    coord = floor_divisor(coord, d_pos + 1);
                coord -= min_c[index_d];
                std::cout<<"("<<coord<<")";
                assert(coord >= 0);
                // Pack the number by shifting it and combining with the packedNumber
                packedNumber |= coord << shift;
                shift = cumulative_num_bits[index_d];
            }
            packedNumber |= r << shift;
            packedNumber |= batch_index << cumulative_num_bits[d_pos];
            std::cout<<"\n\tkey: "<<packedNumber<<"\n";
        }
    }
    return output;
}

size_t hashtable_lattice::test_lattice_point_encoder(const thrust::device_ptr<int> rem0_d, const thrust::device_ptr<int> ranks_d) const
{
    print_vector(cumulative_num_bits_per_dim, "cumulative_num_bits_per_dim");
    print_vector(min_coordinate_per_pos, "min_coordinate_per_pos");
    auto print_std_vector = [](const std::vector<int>& vec, const char* name) {
        std::cout<<name<<": ";
        for (const auto& element : vec) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    };
    std::map<KEY_TYPE, std::pair<VALUE_TYPE, std::vector<int>>> map;
    std::vector<int> rem0(rem0_d, rem0_d + batch_size * num_positions * (d_pos + 1));
    std::vector<int> ranks(ranks_d, ranks_d + batch_size * num_positions * (d_pos + 1));
    std::vector<int> min_c(min_coordinate_per_pos.begin(), min_coordinate_per_pos.end());
    std::vector<int> cumulative_num_bits(cumulative_num_bits_per_dim.begin(), cumulative_num_bits_per_dim.end());
    size_t num_unique_points = 0;
    for(int index_pt = 0; index_pt != batch_size * num_positions; ++index_pt)
    {
        const int batch_index = index_pt / num_positions;
        for (int r = 0; r != d_pos + 1; r++)
        {
            KEY_TYPE packedNumber = 0;
            int shift = 0;
            std::vector<int> current_point;
            for(int index_d = 0; index_d != d_pos; ++index_d)
            {
                const int current_rank = ranks[index_pt * (d_pos + 1) + index_d];
                const int current_rem0 = rem0[index_pt * (d_pos + 1) + index_d];
                int coord = current_rem0 + compute_canonical_simplex_point_coordinate(r, current_rank, d_pos);
                current_point.push_back(coord);
                if (coord != 0)
                    coord = floor_divisor(coord, d_pos + 1);
                coord -= min_c[index_d];
                assert(coord >= 0);
                // Pack the number by shifting it and combining with the packedNumber
                packedNumber |= ((KEY_TYPE) coord) << shift;
                shift = cumulative_num_bits[index_d];
            }
            packedNumber |= ((KEY_TYPE) r) << shift;
            packedNumber |= ((KEY_TYPE) batch_index) << cumulative_num_bits[d_pos];
            const auto [it, success] = map.emplace(packedNumber, std::make_pair(num_unique_points, current_point));
            if(success)
                num_unique_points++;
            else
            {
                // check if pre-existing point has same point coordinates.
                std::vector<int> existing_point = std::get<1>(it->second);
                if (existing_point != current_point)
                {
                    std::cout<<"\nCoordinates mis-match for encoded key: "<<packedNumber<<"\n";
                    print_std_vector(current_point, "current_point");
                    print_std_vector(existing_point, "existing_point");
                    throw std::runtime_error("exiting.");
                }
            }
        }
    }
    return num_unique_points;
}
