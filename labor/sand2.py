import math
import random, copy, numpy as np


def init_input(plaintext, block_size):
    res = [[], [], [], []]
    binary = bin(plaintext)[2:]
    binary = binary.zfill(block_size)
    for i in range(block_size):
        group_index = int(i) % 4
        res[group_index].append(binary[i])
    for r in res:
        r.reverse()
    rl = res[3] + res[2] + res[1] + res[0]
    rl.reverse()
    initial_num = ''.join(rl)
    return int(initial_num, 2)


def g0(num, block_size):
    res = 0
    mask_0 = 0xFF
    mask_1 = 0xFF00
    mask_2 = 0xFF0000
    mask_3 = 0xFF000000
    group_size = block_size // 4
    # y{0} = x{3} and x{2} xor x{0}
    res |= ((num >> (group_size * 3)) & mask_0) & ((num >> (group_size * 2)) & mask_0) ^ (num & mask_0)
    # y{3} = y{0} and x{1} xor x{3}
    res |= ((res << (group_size * 3)) & mask_3) & ((num << (group_size * 2)) & mask_3) ^ (num & mask_3)
    # y{2} = x{2}
    res |= (num & mask_2)
    # y{1} = x{1}
    res |= (num & mask_1)
    return res


def g1(num, block_size):
    res = 0
    mask_0 = 0xFF
    mask_1 = 0xFF00
    mask_2 = 0xFF0000
    mask_3 = 0xFF000000
    group_size = block_size // 4
    # y{2} = x{3} and x{1} xor x{2}
    res |= ((num >> (group_size * 1)) & mask_2) & ((num << (group_size * 1)) & mask_2) ^ (num & mask_2)
    # y{1} = y{2} and x{0} xor x{1}
    res |= ((res >> (group_size * 1)) & mask_1) & ((num << (group_size * 1)) & mask_1) ^ (num & mask_1)
    # y{3} = x{3}
    res |= (num & mask_3)
    # y{0} = x{0}
    res |= (num & mask_0)
    return res


def rotation(num, rot_size, block_size):
    if rot_size == 0:
        return num
    mask_0 = 0xFF
    group_size = block_size // 4
    n_3 = num >> (group_size * 3) & mask_0
    n_2 = num >> (group_size * 2) & mask_0
    n_1 = num >> (group_size * 1) & mask_0
    n_0 = num >> (group_size * 0) & mask_0
    n_3 = (n_3 << rot_size | n_3 >> (group_size - rot_size)) & mask_0

    n_2 = (n_2 << rot_size | n_2 >> (group_size - rot_size)) & mask_0

    n_1 = (n_1 << rot_size | n_1 >> (group_size - rot_size)) & mask_0

    n_0 = (n_0 << rot_size | n_0 >> (group_size - rot_size)) & mask_0

    res = n_3 << group_size * 3 | n_2 << group_size * 2 | n_1 << group_size | n_0
    return res


def a8(num, block_size):
    t0 = 3
    t1 = 1
    step = block_size // 8
    res = num >> step
    x0 = num & 0xF
    x7 = num >> step * 7 & 0xF
    y6 = x7 ^ (x7 << t0)
    y7 = (x7 << t1 | x7 >> (step - t1)) ^ x0
    res = (y7 << step * 7) | (y6 << step * 6) | res
    return res


def key_schedule(key):
    k_3 = init_input(key >> 32 * 3, 32)
    k_2 = init_input(key >> 32 * 2, 32)
    k_1 = init_input(key >> 32 * 1, 32)
    k_0 = init_input(key >> 32 * 0, 32)
    block_size = 128 // 4
    sub_key = [k_0, k_1, k_2, k_3]
    for i in range(48 - 4):
        k = a8(a8(a8(sub_key[i + 3], block_size), block_size), block_size) ^ sub_key[i] ^ (i + 1)
        sub_key.append(k)
    return sub_key


class Sand:
    def __init__(self):
        self.alpha = 0
        self.beta = 1
        self.perm_list = [7, 4, 1, 6, 3, 0, 5, 2]

    def perm(self, num, block_size):
        binary = bin(num)[2:].zfill(block_size)
        binary = [binary[i] for i in range(len(binary))]
        group_size = block_size // 4
        for i in range(4):
            ori = copy.deepcopy(binary[i * group_size:i * group_size + group_size])
            for j in range(group_size):
                binary[i * group_size + j] = ori[self.perm_list[j]]

        return int(''.join(binary), 2)

    def enc(self, plaintext, plaintext2, rounds, word_size, key):
        block_size = word_size // 2
        mask = 0xFFFFFFFF
        sub_key = key_schedule(key)
        if block_size == 64:
            mask = 0xFFFFFFFFFFFFFFFF
        p1_l = plaintext >> block_size
        p1_r = plaintext & mask
        p2_l = plaintext2 >> block_size
        p2_r = plaintext2 & mask
        init1_l = init_input(p1_l, block_size)
        init1_r = init_input(p1_r, block_size)
        init2_l = init_input(p2_l, block_size)
        init2_r = init_input(p2_r, block_size)
        res = 0
        for i in range(rounds):
            ori1l = init1_l
            ori2l = init2_l

            p1_g0 = g0(rotation(init1_l, self.alpha, block_size), block_size)
            p1_g1 = g1(rotation(init1_l, self.beta, block_size), block_size)

            p2_g0 = g0(rotation(init2_l, self.alpha, block_size), block_size)
            p2_g1 = g1(rotation(init2_l, self.beta, block_size), block_size)

            rotl = self.perm(p1_g0 ^ p1_g1, block_size)
            rot2 = self.perm(p2_g0 ^ p2_g1, block_size)

            new_ori1 = rotl ^ init1_r
            new_ori2 = rot2 ^ init2_r

            init1_l = new_ori1
            init2_l = new_ori2
            init1_r = ori1l
            init2_r = ori2l

        c1 = init1_l << block_size | init1_r
        c2 = init2_l << block_size | init2_r
        res = c1 ^ c2
        return res


def print_dff(a):
    r = a & 0xFFFFFFFF
    ll = a >> 32 & 0xFFFFFFFF
    print(hex(ll)[2:].zfill(8))
    print(hex(r)[2:].zfill(8))


def test():
    temp = []
    diff_left = 0x00000004
    diff_right = 0x40040000
    out_left = 0x00010000
    out_right = 0x00000000
    input_dff = diff_left << 32 | diff_right
    output_diff = out_left << 32 | out_right
    total = 2 ** 12
    sand = Sand()
    counter = 0
    key = random.randint(0, 2 ** 128)
    for i in range(total):
        x1 = random.randint(0, 2 ** 64)
        x2 = x1 ^ input_dff
        if x2 < x1:
            i -= 1
            continue
        if x1 in temp:
            i -= 1
            continue
        temp.append(x1)
        df = sand.enc(x1, x2, 2, 64, key)
        if df == output_diff:
            counter += 1
    print(math.log(counter / total, 2))


test()
