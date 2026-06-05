# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Generate test data for hash routing precision comparison.
"""

import numpy as np

HASH_ROUTING_TEST_CASES = [
    (0, 50, 4, 2, "softmax", 1.0, 4, 2, 16),
    (1, 50, 4, 2, "sigmoid", 1.0, 4, 2, 16),
    (2, 50, 4, 2, "sqrtsoftplus", 1.0, 4, 2, 16),
    (3, 50, 4, 2, "softmax", 2.0, 4, 2, 16),
    (4, 50, 4, 2, "sigmoid", 0.5, 4, 2, 16),
    (5, 50, 4, 1, "softmax", 1.0, 4, 2, 16),
    (6, 50, 4, 4, "sqrtsoftplus", 1.0, 4, 2, 16),
    (7, 200, 8, 2, "softmax", 1.0, 8, 2, 16),
    (8, 200, 8, 2, "sigmoid", 1.0, 8, 2, 16),
    (9, 200, 8, 3, "sqrtsoftplus", 1.0, 6, 2, 16),
]


def _case_key(tc):
    return f"v{tc[1]}_e{tc[2]}_k{tc[3]}_{tc[4]}_s{tc[5]:.1f}_l{tc[6]}_b{tc[7]}_h{tc[8]}"


def _make_tid2eid(vocab_size, num_experts, top_k):
    ids = np.arange(vocab_size, dtype=np.int32)
    return np.stack([(ids + k) % num_experts for k in range(top_k)], axis=1)


def _generate_inputs(tc, seed_base=42):
    """Generate deterministic inputs shared by all data types."""
    tid, vocab, experts, topk, score_func, scale, slen, bsz, hidden = tc
    seed = seed_base + tid * 100
    rng = np.random.default_rng(seed)

    hidden_states = (0.01 * rng.standard_normal((slen, bsz, hidden))).astype(np.float32)
    input_ids = rng.integers(0, vocab, size=(bsz, slen)).astype(np.int32)
    weight = (0.01 * rng.standard_normal((experts, hidden))).astype(np.float32)
    tid2eid = _make_tid2eid(vocab, experts, topk)

    return {
        "key": _case_key(tc),
        "hidden_size": hidden,
        "num_experts": experts,
        "top_k": topk,
        "vocab_size": vocab,
        "score_func": score_func,
        "route_scale": scale,
        "seq_length": slen,
        "batch_size": bsz,
        "hidden_states": hidden_states,
        "input_ids": input_ids,
        "weight": weight,
        "tid2eid": tid2eid,
    }


# Build an input data registry
CASE_INPUT_REGISTRY = {}
for _tc in HASH_ROUTING_TEST_CASES:
    _data = _generate_inputs(_tc)
    CASE_INPUT_REGISTRY[_data["key"]] = _data

CASE_KEYS = [CASE_INPUT_REGISTRY[k]["key"] for k in CASE_INPUT_REGISTRY]


def get_golden() -> dict[str, np.ndarray]:
    """Megatron CPU float64 output."""
    return {
        # --- v50_e4_k2_softmax_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k2_softmax_s1.0_l4_b2_h16": np.array([
            [0.25001925, 0.24987277],
            [0.25007933, 0.25002891],
            [0.24995984, 0.25006303],
            [0.25008932, 0.24994712],
            [0.24998394, 0.25012466],
            [0.24993227, 0.25009018],
            [0.25006443, 0.24982177],
            [0.24999133, 0.25000194],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_softmax_s1.0_l4_b2_h16": np.array([
            [3, 0],
            [1, 2],
            [0, 1],
            [1, 2],
            [2, 3],
            [2, 3],
            [2, 3],
            [1, 2],
        ], dtype=np.int64),

        # --- v50_e4_k2_sigmoid_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k2_sigmoid_s1.0_l4_b2_h16": np.array([
            [0.500072, 0.49992797],
            [0.50001967, 0.49998036],
            [0.4999277, 0.5000723],
            [0.49997616, 0.50002384],
            [0.49997616, 0.50002384],
            [0.49999103, 0.50000894],
            [0.49996343, 0.50003654],
            [0.49997571, 0.50002426],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_sigmoid_s1.0_l4_b2_h16": np.array([
            [3, 0],
            [2, 3],
            [1, 2],
            [0, 1],
            [3, 0],
            [1, 2],
            [3, 0],
            [2, 3],
        ], dtype=np.int64),

        # --- v50_e4_k2_sqrtsoftplus_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k2_sqrtsoftplus_s1.0_l4_b2_h16": np.array([
            [0.49993262, 0.50006735],
            [0.49995571, 0.50004429],
            [0.50007629, 0.49992368],
            [0.5000295, 0.49997053],
            [0.49995777, 0.50004226],
            [0.49994418, 0.50005585],
            [0.50008351, 0.49991649],
            [0.49995333, 0.50004667],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_sqrtsoftplus_s1.0_l4_b2_h16": np.array([
            [1, 2],
            [3, 0],
            [3, 0],
            [3, 0],
            [1, 2],
            [3, 0],
            [0, 1],
            [0, 1],
        ], dtype=np.int64),

        # --- v50_e4_k2_softmax_s2.0_l4_b2_h16 ---
        "top_scores_v50_e4_k2_softmax_s2.0_l4_b2_h16": np.array([
            [0.49963182, 0.49998271],
            [0.49994996, 0.4999148],
            [0.49985814, 0.5003224],
            [0.49994346, 0.50028211],
            [0.4997187, 0.50003827],
            [0.49988562, 0.50014061],
            [0.49990606, 0.50024891],
            [0.49999073, 0.49985951],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_softmax_s2.0_l4_b2_h16": np.array([
            [3, 0],
            [1, 2],
            [0, 1],
            [1, 2],
            [2, 3],
            [1, 2],
            [3, 0],
            [2, 3],
        ], dtype=np.int64),

        # --- v50_e4_k2_sigmoid_s0.5_l4_b2_h16 ---
        "top_scores_v50_e4_k2_sigmoid_s0.5_l4_b2_h16": np.array([
            [0.25000471, 0.24999529],
            [0.25002977, 0.24997023],
            [0.25002426, 0.24997574],
            [0.25005591, 0.24994409],
            [0.24999022, 0.25000978],
            [0.2500138, 0.24998619],
            [0.24999441, 0.25000557],
            [0.25001529, 0.24998471],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_sigmoid_s0.5_l4_b2_h16": np.array([
            [0, 1],
            [3, 0],
            [2, 3],
            [2, 3],
            [2, 3],
            [1, 2],
            [0, 1],
            [1, 2],
        ], dtype=np.int64),

        # --- v50_e4_k1_softmax_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k1_softmax_s1.0_l4_b2_h16": np.array([
            [0.24989754],
            [0.25008219],
            [0.24994154],
            [0.24997066],
            [0.24989815],
            [0.25016347],
            [0.24995331],
            [0.24991991],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k1_softmax_s1.0_l4_b2_h16": np.array([
            [1],
            [0],
            [2],
            [1],
            [3],
            [0],
            [3],
            [1],
        ], dtype=np.int64),

        # --- v50_e4_k4_sqrtsoftplus_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k4_sqrtsoftplus_s1.0_l4_b2_h16": np.array([
            [0.24999793, 0.2499841, 0.25004816, 0.2499698],
            [0.25001055, 0.24999355, 0.24999005, 0.25000587],
            [0.25000274, 0.25003248, 0.24996062, 0.25000417],
            [0.24997903, 0.25001973, 0.2500093, 0.24999192],
            [0.24998823, 0.24998194, 0.24999522, 0.25003463],
            [0.24997857, 0.25001818, 0.25004983, 0.24995343],
            [0.24994256, 0.25001538, 0.24997631, 0.25006577],
            [0.24995397, 0.2500954, 0.24995506, 0.24999556],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k4_sqrtsoftplus_s1.0_l4_b2_h16": np.array([
            [2, 3, 0, 1],
            [2, 3, 0, 1],
            [1, 2, 3, 0],
            [0, 1, 2, 3],
            [2, 3, 0, 1],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [0, 1, 2, 3],
        ], dtype=np.int64),

        # --- v200_e8_k2_softmax_s1.0_l8_b2_h16 ---
        "top_scores_v200_e8_k2_softmax_s1.0_l8_b2_h16": np.array([
            [0.12497517, 0.12504654],
            [0.12497249, 0.12497474],
            [0.12500283, 0.12502088],
            [0.12506332, 0.12507972],
            [0.12492646, 0.1249902],
            [0.12499807, 0.12498959],
            [0.12497015, 0.12515196],
            [0.12497307, 0.12507308],
            [0.12495933, 0.1250089],
            [0.12503949, 0.12495708],
            [0.12494654, 0.12498458],
            [0.12504083, 0.12499334],
            [0.1249186, 0.12498166],
            [0.12499966, 0.12500179],
            [0.12496792, 0.1250723],
            [0.12503614, 0.12503219],
        ], dtype=np.float32),
        "selected_experts_indices_v200_e8_k2_softmax_s1.0_l8_b2_h16": np.array([
            [3, 4],
            [6, 7],
            [1, 2],
            [1, 2],
            [5, 6],
            [7, 0],
            [7, 0],
            [5, 6],
            [2, 3],
            [2, 3],
            [2, 3],
            [3, 4],
            [0, 1],
            [2, 3],
            [7, 0],
            [7, 0],
        ], dtype=np.int64),

        # --- v200_e8_k2_sigmoid_s1.0_l8_b2_h16 ---
        "top_scores_v200_e8_k2_sigmoid_s1.0_l8_b2_h16": np.array([
            [0.49999556, 0.50000447],
            [0.49993041, 0.50006962],
            [0.50000733, 0.49999267],
            [0.50000799, 0.49999204],
            [0.50002265, 0.49997732],
            [0.5000242, 0.49997577],
            [0.49992424, 0.50007576],
            [0.49989706, 0.50010294],
            [0.49996442, 0.50003558],
            [0.49998307, 0.50001693],
            [0.50009811, 0.49990186],
            [0.4999595, 0.50004047],
            [0.49996549, 0.50003451],
            [0.50010437, 0.49989563],
            [0.50004137, 0.49995863],
            [0.49998102, 0.50001895],
        ], dtype=np.float32),
        "selected_experts_indices_v200_e8_k2_sigmoid_s1.0_l8_b2_h16": np.array([
            [3, 4],
            [5, 6],
            [2, 3],
            [5, 6],
            [6, 7],
            [6, 7],
            [2, 3],
            [0, 1],
            [4, 5],
            [3, 4],
            [1, 2],
            [1, 2],
            [1, 2],
            [4, 5],
            [0, 1],
            [3, 4],
        ], dtype=np.int64),

        # --- v200_e8_k3_sqrtsoftplus_s1.0_l6_b2_h16 ---
        "top_scores_v200_e8_k3_sqrtsoftplus_s1.0_l6_b2_h16": np.array([
            [0.3333216, 0.33339602, 0.33328235],
            [0.33335274, 0.33330044, 0.33334681],
            [0.33333218, 0.33328143, 0.33338639],
            [0.3333703, 0.33333015, 0.33329955],
            [0.33330098, 0.33337143, 0.33332756],
            [0.33334315, 0.33333737, 0.33331949],
            [0.33334586, 0.3333855, 0.33326864],
            [0.33334172, 0.33332828, 0.33333001],
            [0.33330792, 0.33340326, 0.33328882],
            [0.33337891, 0.33334175, 0.33327931],
            [0.33334443, 0.3333005, 0.33335504],
            [0.33332208, 0.33330595, 0.33337194],
        ], dtype=np.float32),
        "selected_experts_indices_v200_e8_k3_sqrtsoftplus_s1.0_l6_b2_h16": np.array([
            [6, 7, 0],
            [6, 7, 0],
            [7, 0, 1],
            [4, 5, 6],
            [7, 0, 1],
            [5, 6, 7],
            [5, 6, 7],
            [0, 1, 2],
            [0, 1, 2],
            [1, 2, 3],
            [4, 5, 6],
            [3, 4, 5],
        ], dtype=np.int64),
    }
    # ===================================================


def get_gpu_datas() -> dict[str, np.ndarray]:
    """Megatron GPU float32 output."""
    return {
        # --- v50_e4_k2_softmax_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k2_softmax_s1.0_l4_b2_h16": np.array([
            [0.24987274, 0.25001928],
            [0.25007936, 0.25002891],
            [0.24995978, 0.25006303],
            [0.25008932, 0.24994712],
            [0.24998397, 0.25012466],
            [0.24993224, 0.25009024],
            [0.25006443, 0.24982175],
            [0.24999131, 0.25000194],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_softmax_s1.0_l4_b2_h16": np.array([
            [0, 3],
            [1, 2],
            [0, 1],
            [1, 2],
            [2, 3],
            [2, 3],
            [2, 3],
            [1, 2],
        ], dtype=np.int64),

        # --- v50_e4_k2_sigmoid_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k2_sigmoid_s1.0_l4_b2_h16": np.array([
            [0.49992794, 0.500072],
            [0.50001961, 0.49998039],
            [0.49992773, 0.50007224],
            [0.49997616, 0.50002384],
            [0.50002384, 0.49997616],
            [0.499991, 0.500009],
            [0.5000366, 0.49996346],
            [0.49997574, 0.50002432],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_sigmoid_s1.0_l4_b2_h16": np.array([
            [0, 3],
            [2, 3],
            [1, 2],
            [0, 1],
            [0, 3],
            [1, 2],
            [0, 3],
            [2, 3],
        ], dtype=np.int64),

        # --- v50_e4_k2_sqrtsoftplus_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k2_sqrtsoftplus_s1.0_l4_b2_h16": np.array([
            [0.49993268, 0.50006735],
            [0.50004429, 0.49995571],
            [0.49992368, 0.50007629],
            [0.4999705, 0.5000295],
            [0.49995777, 0.50004226],
            [0.50005585, 0.49994415],
            [0.50008357, 0.49991646],
            [0.49995333, 0.50004667],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_sqrtsoftplus_s1.0_l4_b2_h16": np.array([
            [1, 2],
            [0, 3],
            [0, 3],
            [0, 3],
            [1, 2],
            [0, 3],
            [0, 1],
            [0, 1],
        ], dtype=np.int64),

        # --- v50_e4_k2_softmax_s2.0_l4_b2_h16 ---
        "top_scores_v50_e4_k2_softmax_s2.0_l4_b2_h16": np.array([
            [0.49998268, 0.49963185],
            [0.49994996, 0.4999148],
            [0.49985808, 0.50032234],
            [0.49994346, 0.50028217],
            [0.49971879, 0.50003833],
            [0.49988562, 0.50014055],
            [0.50024891, 0.49990606],
            [0.49999079, 0.49985948],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_softmax_s2.0_l4_b2_h16": np.array([
            [0, 3],
            [1, 2],
            [0, 1],
            [1, 2],
            [2, 3],
            [1, 2],
            [0, 3],
            [2, 3],
        ], dtype=np.int64),

        # --- v50_e4_k2_sigmoid_s0.5_l4_b2_h16 ---
        "top_scores_v50_e4_k2_sigmoid_s0.5_l4_b2_h16": np.array([
            [0.25000471, 0.24999529],
            [0.24997023, 0.25002977],
            [0.25002423, 0.24997574],
            [0.25005591, 0.24994411],
            [0.24999021, 0.25000978],
            [0.2500138, 0.2499862],
            [0.24999443, 0.2500056],
            [0.25001526, 0.24998474],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k2_sigmoid_s0.5_l4_b2_h16": np.array([
            [0, 1],
            [0, 3],
            [2, 3],
            [2, 3],
            [2, 3],
            [1, 2],
            [0, 1],
            [1, 2],
        ], dtype=np.int64),

        # --- v50_e4_k1_softmax_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k1_softmax_s1.0_l4_b2_h16": np.array([
            [0.24989754],
            [0.25008219],
            [0.24994156],
            [0.24997069],
            [0.24989817],
            [0.25016344],
            [0.24995328],
            [0.24991992],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k1_softmax_s1.0_l4_b2_h16": np.array([
            [1],
            [0],
            [2],
            [1],
            [3],
            [0],
            [3],
            [1],
        ], dtype=np.int64),

        # --- v50_e4_k4_sqrtsoftplus_s1.0_l4_b2_h16 ---
        "top_scores_v50_e4_k4_sqrtsoftplus_s1.0_l4_b2_h16": np.array([
            [0.25004819, 0.24996981, 0.24999793, 0.2499841],
            [0.24999005, 0.25000584, 0.25001055, 0.24999356],
            [0.25000417, 0.25000274, 0.25003248, 0.24996062],
            [0.249979, 0.25001973, 0.25000933, 0.24999194],
            [0.2499952, 0.2500346, 0.24998823, 0.24998194],
            [0.24995345, 0.24997857, 0.25001818, 0.25004983],
            [0.24997629, 0.25006574, 0.24994256, 0.25001538],
            [0.24995402, 0.25009537, 0.24995507, 0.24999556],
        ], dtype=np.float32),
        "selected_experts_indices_v50_e4_k4_sqrtsoftplus_s1.0_l4_b2_h16": np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ], dtype=np.int64),

        # --- v200_e8_k2_softmax_s1.0_l8_b2_h16 ---
        "top_scores_v200_e8_k2_softmax_s1.0_l8_b2_h16": np.array([
            [0.12497517, 0.12504654],
            [0.1249725, 0.12497471],
            [0.12500283, 0.12502086],
            [0.1250633, 0.12507974],
            [0.12492647, 0.1249902],
            [0.1249896, 0.12499808],
            [0.12515195, 0.12497013],
            [0.12497307, 0.12507308],
            [0.12495934, 0.1250089],
            [0.12503949, 0.12495708],
            [0.12494652, 0.12498459],
            [0.12504083, 0.12499335],
            [0.12491859, 0.12498167],
            [0.12499965, 0.12500177],
            [0.12507229, 0.1249679],
            [0.12503217, 0.12503614],
        ], dtype=np.float32),
        "selected_experts_indices_v200_e8_k2_softmax_s1.0_l8_b2_h16": np.array([
            [3, 4],
            [6, 7],
            [1, 2],
            [1, 2],
            [5, 6],
            [0, 7],
            [0, 7],
            [5, 6],
            [2, 3],
            [2, 3],
            [2, 3],
            [3, 4],
            [0, 1],
            [2, 3],
            [0, 7],
            [0, 7],
        ], dtype=np.int64),

        # --- v200_e8_k2_sigmoid_s1.0_l8_b2_h16 ---
        "top_scores_v200_e8_k2_sigmoid_s1.0_l8_b2_h16": np.array([
            [0.49999553, 0.50000447],
            [0.49993041, 0.50006962],
            [0.50000739, 0.49999267],
            [0.50000799, 0.49999204],
            [0.50002271, 0.49997732],
            [0.50002426, 0.49997574],
            [0.49992421, 0.50007576],
            [0.49989706, 0.50010294],
            [0.49996442, 0.50003558],
            [0.49998313, 0.50001693],
            [0.50009811, 0.49990192],
            [0.49995956, 0.50004047],
            [0.49996546, 0.50003457],
            [0.50010437, 0.49989566],
            [0.50004131, 0.49995863],
            [0.49998099, 0.50001901],
        ], dtype=np.float32),
        "selected_experts_indices_v200_e8_k2_sigmoid_s1.0_l8_b2_h16": np.array([
            [3, 4],
            [5, 6],
            [2, 3],
            [5, 6],
            [6, 7],
            [6, 7],
            [2, 3],
            [0, 1],
            [4, 5],
            [3, 4],
            [1, 2],
            [1, 2],
            [1, 2],
            [4, 5],
            [0, 1],
            [3, 4],
        ], dtype=np.int64),

        # --- v200_e8_k3_sqrtsoftplus_s1.0_l6_b2_h16 ---
        "top_scores_v200_e8_k3_sqrtsoftplus_s1.0_l6_b2_h16": np.array([
            [0.33328235, 0.3333216, 0.33339602],
            [0.33334681, 0.33335271, 0.33330044],
            [0.33328143, 0.33338639, 0.33333215],
            [0.3333703, 0.33333018, 0.33329955],
            [0.33337146, 0.33332756, 0.33330098],
            [0.33334315, 0.33333737, 0.33331949],
            [0.33334589, 0.33338544, 0.33326861],
            [0.33334169, 0.33332828, 0.33333001],
            [0.33330792, 0.33340326, 0.33328882],
            [0.33337891, 0.33334178, 0.33327931],
            [0.33334443, 0.33330053, 0.33335507],
            [0.33332211, 0.33330598, 0.33337194],
        ], dtype=np.float32),
        "selected_experts_indices_v200_e8_k3_sqrtsoftplus_s1.0_l6_b2_h16": np.array([
            [0, 6, 7],
            [0, 6, 7],
            [0, 1, 7],
            [4, 5, 6],
            [0, 1, 7],
            [5, 6, 7],
            [5, 6, 7],
            [0, 1, 2],
            [0, 1, 2],
            [1, 2, 3],
            [4, 5, 6],
            [3, 4, 5],
        ], dtype=np.int64),
    }
    # ===================================================


GOLDEN_DATA = get_golden()
GPU_DATA = get_gpu_datas()

if __name__ == "__main__":
    print(f"Test case keys ({len(CASE_KEYS)}):")
    for k in CASE_KEYS:
        info = CASE_INPUT_REGISTRY[k]
        print(f"  {k}")
        print(f"    score_func={info['score_func']}, route_scale={info['route_scale']}, "
              f"top_k={info['top_k']}, experts={info['num_experts']}, vocab={info['vocab_size']}, "
              f"seq={info['seq_length']}x{info['batch_size']}, hidden={info['hidden_size']}")

    golden_keys = list(GOLDEN_DATA.keys())
    gpu_keys = list(GPU_DATA.keys())
    print(f"\nGOLDEN_DATA keys: {len(golden_keys)} entries")
    print(f"GPU_DATA keys:    {len(gpu_keys)} entries")
