import katan_self
import util
import and_bct
import random

L1_PROBABILITY = 1  # the difference pair in bct with some probability
L2_PROBABILITY = 1

TEST_ROUND = 4

L1_INDEXES_ORI = [[5, 8], [4, 7], [3, 6], [2, 5]]
L1_INDEXES_ORI_OUT = [[e + TEST_ROUND for e in n] for n in L1_INDEXES_ORI]

L2_INDEXES_ORI = [[3, 8, 10, 12], [2, 7, 9, 11], [1, 6, 8, 10], [0, 5, 7, 9]]
L2_INDEXES_ORI_OUT = [[e + TEST_ROUND for e in n] for n in L2_INDEXES_ORI]

L1_RIGHT = [12 - i for i in range(TEST_ROUND)]
L2_RIGHT = [18 - i for i in range(TEST_ROUND)]


def bct_tool(bct):
    # List the probabilities of 1 and 0 for BCT
    size = len(bct[0])
    p_0 = {}
    p_1 = {}
    for i in range(size):
        for j in range(size):
            if bct[i][j] == size:
                if i not in p_1:
                    p_1[i] = [j]
                else:
                    p_1[i].append(j)
            else:
                if i not in p_0:
                    p_0[i] = [j]
                else:
                    p_0[i].append(j)
    return p_0, p_1


def find_appropriate_differential(bct_p0, bct_p1, indexes_ori, indexes_ori_out, target_p):
    list_size = len(indexes_ori)
    size = len(indexes_ori[0])
    diff_pair = []
    log_in_map = {}
    log_out_map = {}
    flag = False
    while not flag:
        log_in_map = {}
        log_out_map = {}
        diff_pair = []
        flag = True
        for i in range(list_size):
            _in_indexes = indexes_ori[i]
            bits = []
            for index in _in_indexes:
                bit = random.randint(0, 1)
                if index in log_in_map:
                    bit = log_in_map[index]
                else:
                    log_in_map[index] = bit
                bits.append(bit)
            _in = util.bits_to_num(bits)
            _out_list = bct_p1[_in]
            _out = bct_p1[_in][random.randint(0, len(_out_list) - 1)]
            if target_p == 0:
                if _in in bct_p0:
                    _out = bct_p0[_in][0]
                else:
                    flag = False
                    break
            _out_bits = util.num_to_bits(_out, size)
            _out_indexes = indexes_ori_out[i]
            for j in range(size):
                k = _out_indexes[j]
                if k not in log_out_map:
                    log_out_map[k] = _out_bits[j]
                else:
                    val = log_out_map[k]
                    if val != _out_bits[j]:
                        flag = False
            if not flag:
                break
            diff_pair.append([_in, _out])
    return log_in_map, log_out_map, diff_pair


def verify_differ_pair(log_in_map, log_out_map, diff_pair, indexes_ori, indexes_ori_out):
    size = len(indexes_ori)
    for i in range(size):
        in_indexes = indexes_ori[i]
        _in_bits = []
        for j in in_indexes:
            _in_bits.append(log_in_map[j])
        out_indexes = indexes_ori_out[i]
        _out_bits = []
        for j in out_indexes:
            _out_bits.append(log_out_map[j])
        _in = util.bits_to_num(_in_bits)
        _out = util.bits_to_num(_out_bits)
        diff = diff_pair[i]
        assert diff[0] == _in
        if diff[1] != _out:
            print()
        assert diff[1] == _out


def verify_multi_round_switch():
    l1_bct = and_bct.create_and_bct(and_bct.general_and_operation, 1)
    l2_bct = and_bct.create_and_bct(and_bct.general_and_xor_connection, 2)

    l1_p0, l1_p1 = bct_tool(l1_bct)
    l2_p0, l2_p1 = bct_tool(l2_bct)

    l1_in_map, l1_out_map, l1_diff = find_appropriate_differential(l1_p0, l1_p1, L1_INDEXES_ORI, L1_INDEXES_ORI_OUT,
                                                                   L1_PROBABILITY)
    l2_in_map, l2_out_map, l2_diff = find_appropriate_differential(l2_p0, l2_p1, L2_INDEXES_ORI, L2_INDEXES_ORI_OUT,
                                                                   L2_PROBABILITY)

    verify_differ_pair(l1_in_map, l1_out_map, l1_diff, L1_INDEXES_ORI, L1_INDEXES_ORI_OUT)
    verify_differ_pair(l2_in_map, l2_out_map, l2_diff, L2_INDEXES_ORI, L2_INDEXES_ORI_OUT)

    # verify
    key = random.randint(0, 2 ** 32 - 1)
    l1_right_diff = random.randint(0, 2 ** 4 - 1)
    l2_right_diff = random.randint(0, 2 ** 4 - 1)
    counter = 0
    cases = 10000
    p = L1_PROBABILITY * L2_PROBABILITY
    for i in range(cases):
        x1 = random.randint(0, 2 ** 32 - 1)
        x1_bits = util.num_to_bits(x1)
        x1_l1, x1_l2 = x1_bits[19:], x1_bits[:19]
        x2_l1, x2_l2 = compute_diff(x1_l1, l1_in_map), compute_diff(x1_l2, l2_in_map)
        x2_l1 = compute_diff2(x2_l1, l1_right_diff, L1_RIGHT)
        x2_l2 = compute_diff2(x2_l2, l2_right_diff, L2_RIGHT)
        c1_l1, c1_l2 = katan_self.enc32_bit(x1_l1, x1_l2, key, 4)
        c2_l1, c2_l2 = katan_self.enc32_bit(x2_l1, x2_l2, key, 4)
        c3_l1, c3_l2 = compute_diff(c1_l1, l1_out_map), compute_diff(c1_l2, l2_out_map)
        c4_l1, c4_l2 = compute_diff(c2_l1, l1_out_map), compute_diff(c2_l2, l2_out_map)
        x3_l1, x3_l2 = katan_self.dec32_bit(c3_l1, c3_l2, key, 4)
        x4_l1, x4_l2 = katan_self.dec32_bit(c4_l1, c4_l2, key, 4)

        diff_l1 = get_difference(x3_l1, x4_l1, L1_RIGHT)
        diff_l2 = get_difference(x3_l2, x4_l2, L2_RIGHT)
        if diff_l1 == l1_right_diff and diff_l2 == l2_right_diff:
            counter += 1
    if p == 1:
        assert counter == cases
    else:
        assert counter == 0


def compute_diff(bits, diff_map):
    n_bits = bits[:]
    for k in diff_map:
        n_bits[k] = bits[k] ^ diff_map[k]
    return n_bits


def compute_diff2(register: list, delta, bits_indexes):
    new_reg = register[:]
    delta_bits = util.num_to_bits(delta, len(bits_indexes))
    for i in range(len(bits_indexes)):
        index = bits_indexes[i]
        new_reg[index] = register[index] ^ delta_bits[i]
    return new_reg


def get_difference(reg1, reg2, bits_indexes):
    n_res = []
    for ind in bits_indexes:
        n_res.append(reg1[ind] ^ reg2[ind])
    return util.bits_to_num(n_res)


def get_left_part_differ_list(x1, x2, l1_indexes, l2_indexes):
    x1_bits = util.num_to_bits(x1)
    x1_l1, x1_l2 = x1_bits[19:], x1_bits[:19]
    x2_bits = util.num_to_bits(x2)
    x2_l1, x2_l2 = x2_bits[19:], x2_bits[:19]
    l1_diff, l2_diff = [], []
    for indexes in l1_indexes:
        bits = []
        for i in indexes:
            bits.append(x1_l1[i] ^ x2_l1[i])
        _diff = util.bits_to_num(bits)
        l1_diff.append(_diff)
    for indexes in l2_indexes:
        bits = []
        for i in indexes:
            bits.append(x1_l2[i] ^ x2_l2[i])
        _diff = util.bits_to_num(bits)
        l2_diff.append(_diff)
    return l1_diff, l2_diff


for _ in range(100):
    print("-----TEST:{0}----".format(_))
    verify_multi_round_switch()
    print("success")
