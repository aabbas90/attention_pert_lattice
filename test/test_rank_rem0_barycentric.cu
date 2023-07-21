#include <vector>
#include <thrust/device_vector.h>
#include "attention_pert_lattice.h"

int main(int argc, char** argv)
{

    const int num_positions = 8;
    const int d_pos = 3;

    std::vector<int> rem0_h(32, 0);
    thrust::device_vector<int> expected_rem0(32, 0);
    std::vector<int> expected_ranks_h = {
        1, 0, 3, 2,
        3, 2, 0, 1,
        1, 3, 0, 2,
        3, 1, 0, 2,
        2, 1, 0, 3,
        1, 2, 0, 3,
        3, 0, 1, 2,
        0, 1, 2, 3};

    std::vector<float> expected_barycentric_h = {
        0.5060, 0.0072, 0.1222, 0.3646,
        0.5251, 0.1935, 0.1180, 0.1635,
        0.4834, 0.2307, 0.0741, 0.2118,
        0.6012, 0.0773, 0.0893, 0.2322,
        0.7732, 0.0350, 0.0750, 0.1168,
        0.6952, 0.1047, 0.1350, 0.0651,
        0.6283, 0.1750, 0.1741, 0.0226,
        0.5153, 0.2673, 0.1422, 0.0752};

    std::vector<float> feat_h = {
        -0.1130,  1.3455, -0.6306, -0.6018,
        -0.9799, -0.2060,  0.9199,  0.2660,
         0.1671, -1.0521,  1.0143, -0.1294,
        -0.6426,  0.0238,  0.9525, -0.3336,
        -0.2318,  0.0682,  0.5354, -0.3718,
         0.3096, -0.2304,  0.5699, -0.6492,
        -0.8958,  0.5909,  0.5006, -0.1958,
         0.7774,  0.4764, -0.0923, -1.1614};

    thrust::device_vector<float> feat(feat_h.begin(), feat_h.end());
    thrust::device_vector<int> rem0(feat.size());
    thrust::device_vector<int> ranks(feat.size());
    thrust::device_vector<float> barycentric(feat.size());
    compute_rem0_rank_barycentric<float>(num_positions, d_pos + 1,
        feat.data(), rem0.data(), ranks.data(), barycentric.data());
    print_matrix(rem0, "rem0", d_pos + 1);
    print_matrix(ranks, "ranks", d_pos + 1);
    print_matrix(barycentric, "barycentric", d_pos + 1);
}
