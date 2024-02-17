import math

import numpy
from numba import cuda
import random
import time
import operator
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

INPUT_LEFT_DIFF = 0x20000000
INPUT_RIGHT_DIFF = 0x21000000
OUTPUT_LEFT_DIFF = 0x00000000
OUTPUT_RIGHT_DIFF = 0x00000000
WORD_SIZE = 64
THREADS_IN_PER_BLOCK_EXPO = 10
BLOCK_IN_PER_GRID_EXPO = 15
TASK_NUM_IN_PER_THREAD_EXPO = 32 - THREADS_IN_PER_BLOCK_EXPO - BLOCK_IN_PER_GRID_EXPO
ROUNDS = 5


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


# #@cuda.jit
# def init_input_gpu(plaintext, block_size, temp_list):
# res = [[], [], [], []]
#
# binary = bin(plaintext)[2:]
# binary = binary.zfill(block_size)
# for i in range(block_size):
#     group_index = int(i) % 4
#     res[group_index].append(binary[i])
# for r in res:
#     r.reverse()
# rl = res[3] + res[2] + res[1] + res[0]
# rl.reverse()
# initial_num = ''.join(rl)
#
# temp_list[1] = int(initial_num, 2)


@cuda.jit(device=True)
def g0(num, block_size):
    group_size = block_size // 4
    x0 = operator.and_(num, 0xFF)
    x1 = operator.and_(operator.rshift(num, group_size), 0xFF)
    x2 = operator.and_(operator.rshift(num, group_size * 2), 0xFF)
    x3 = operator.and_(operator.rshift(num, group_size * 3), 0xFF)

    # y{0} = x{3} and x{2} xor x{0}
    y0 = operator.xor(operator.and_(x3, x2), x0)
    # y{3} = y{0} and x{1} xor x{3}
    y3 = operator.xor(operator.and_(y0, x1), x3)
    # y{2} = x{2}
    y2 = x2
    # y{1} = x{1}
    y1 = x1

    return operator.xor(
        operator.xor(
            operator.xor(
                operator.lshift(y3, group_size * 3),
                operator.lshift(y2, group_size * 2),
            ),
            operator.lshift(y1, group_size)
        ), y0
    )


@cuda.jit(device=True)
def g1(num, block_size):
    group_size = block_size // 4
    x0 = operator.and_(num, 0xFF)
    x1 = operator.and_(operator.rshift(num, group_size), 0xFF)
    x2 = operator.and_(operator.rshift(num, group_size * 2), 0xFF)
    x3 = operator.and_(operator.rshift(num, group_size * 3), 0xFF)

    # y{2} = x{3} and x{1} xor x{2}
    y2 = operator.xor(operator.and_(x3, x1), x2)
    # y{1} = y{2} and x{0} xor x{1}
    y1 = operator.xor(operator.and_(y2, x0), x1)
    # y{3} = x{3}
    y3 = x3
    # y{0} = x{0}
    y0 = x0
    return operator.xor(
        operator.xor(
            operator.xor(
                operator.lshift(y3, group_size * 3),
                operator.lshift(y2, group_size * 2),
            ),
            operator.lshift(y1, group_size)
        ), y0
    )


@cuda.jit(device=True)
def rotation(num, rot_size, block_size):
    if rot_size > 0:
        group_size = block_size // 4
        n3 = operator.and_(operator.rshift(num, group_size * 3), 0xFF)
        n2 = operator.and_(operator.rshift(num, group_size * 2), 0xFF)
        n1 = operator.and_(operator.rshift(num, group_size * 1), 0xFF)
        n0 = operator.and_(num, 0xFF)

        n3 = operator.and_(operator.xor(operator.lshift(n3, rot_size), operator.rshift(n3, group_size - rot_size)),
                           0xFF)
        n2 = operator.and_(operator.xor(operator.lshift(n2, rot_size), operator.rshift(n2, group_size - rot_size)),
                           0xFF)
        n1 = operator.and_(operator.xor(operator.lshift(n1, rot_size), operator.rshift(n1, group_size - rot_size)),
                           0xFF)
        n0 = operator.and_(operator.xor(operator.lshift(n0, rot_size), operator.rshift(n0, group_size - rot_size)),
                           0xFF)

        return operator.xor(
            operator.xor(
                operator.xor(
                    operator.lshift(n3, group_size * 3),
                    operator.lshift(n2, group_size * 2),
                ),
                operator.lshift(n1, group_size)
            ), n0
        )
    else:
        return num


@cuda.jit(device=True)
def perm(num, block_size, perm_list):
    group_size = block_size // 4

    res0 = operator.and_(operator.rshift(operator.and_(num, 0xFF), 7 - perm_list[0]), 0b1)
    res1 = operator.and_(operator.rshift(operator.and_(operator.rshift(num, group_size), 0xFF), 7 - perm_list[0]), 0b1)
    res2 = operator.and_(operator.rshift(operator.and_(operator.rshift(num, group_size * 2), 0xFF), 7 - perm_list[0]),
                         0b1)
    res3 = operator.and_(operator.rshift(operator.and_(operator.rshift(num, group_size * 3), 0xFF), 7 - perm_list[0]),
                         0b1)

    for j in range(1, group_size):
        res0 = operator.lshift(res0, 1)
        res0 = operator.xor(res0, operator.and_(operator.rshift(operator.and_(num, 0xFF), 7 - perm_list[j]), 0b1))
    for j in range(1, group_size):
        res1 = operator.lshift(res1, 1)
        res1 = operator.xor(res1, operator.and_(
            operator.rshift(operator.and_(operator.rshift(num, group_size), 0xFF), 7 - perm_list[j]), 0b1))
    for j in range(1, group_size):
        res2 = operator.lshift(res2, 1)
        res2 = operator.xor(res2, operator.and_(
            operator.rshift(operator.and_(operator.rshift(num, group_size * 2), 0xFF), 7 - perm_list[j]),
            0b1))
    for j in range(1, group_size):
        res3 = operator.lshift(res3, 1)
        res3 = operator.xor(res3, operator.and_(
            operator.rshift(operator.and_(operator.rshift(num, group_size * 3), 0xFF), 7 - perm_list[j]),
            0b1))

    return operator.xor(
        operator.xor(
            operator.xor(
                operator.lshift(res3, group_size * 3),
                operator.lshift(res2, group_size * 2),
            ),
            operator.lshift(res1, group_size)
        ), res0
    )


@cuda.jit(device=True)
def enc(plaintext, rounds, word_size, keys, perm_list):
    alpha = 0
    beta = 1
    block_size = numpy.uint32(word_size // 2)
    p_l = operator.rshift(plaintext, block_size)
    p_r = operator.and_(plaintext, 0xFFFFFFFF)
    # init_input_gpu(p_l, block_size, temp_list)
    init_l = numpy.uint32(p_l)
    # init_input_gpu(p_r, block_size, temp_list)
    init_r = numpy.uint32(p_r)
    for i in range(rounds):
        ori_l = init_l

        rot_g0 = rotation(init_l, alpha, block_size)

        g0_out = g0(rot_g0, block_size)

        rot_g1 = rotation(init_l, beta, block_size)

        g1_out = g1(rot_g1, block_size)

        perm_out = perm(operator.xor(g0_out, g1_out), block_size, perm_list)

        init_l = operator.xor(perm_out, init_r)
        init_r = ori_l

        init_l = numpy.uint32(init_l)
    return init_l << block_size | init_r


@cuda.jit(device=True)
def dec(ciphertext, rounds, word_size, keys, perm_list):
    alpha = 0
    beta = 1
    block_size = numpy.uint32(word_size // 2)
    c_l = operator.rshift(ciphertext, block_size)
    c_r = operator.and_(ciphertext, 0xFFFFFFFF)
    # init_input_gpu(p_l, block_size, temp_list)
    init_l = numpy.uint32(c_l)
    # init_input_gpu(p_r, block_size, temp_list)
    init_r = numpy.uint32(c_r)
    for i in range(rounds - 1, -1, -1):
        ori_r = init_r

        rot_g0 = rotation(init_l, alpha, block_size)

        g0_out = g0(rot_g0, block_size)

        rot_g1 = rotation(init_l, beta, block_size)

        g1_out = g1(rot_g1, block_size)

        perm_out = perm(operator.xor(g0_out, g1_out), block_size, perm_list)

        init_r = operator.xor(perm_out, init_l)

        init_l = ori_r

        init_r = numpy.uint32(init_r)
    return init_l << block_size | init_r


@cuda.jit
def start_gpu_task(keys, input_diff, output_diff, rounds, result_collector, word_size, perm_list, task_num, rng_states):
    thread_index = operator.add(operator.mul(cuda.blockIdx.x, cuda.blockDim.x), cuda.threadIdx.x)
    res = 0
    start = thread_index * (2 ** task_num)
    end = (thread_index + 1) * (2 ** task_num)
    # random_vari = 1
    # if thread_index % 11 == 0:
    #     random_vari = 2 ** 32
    for i in range(start, end):
        x = xoroshiro128p_uniform_float32(rng_states, thread_index)
        random_vari = numpy.uint64(1)
        if x >= 0.5:
            random_vari = 2 ** 32
        x1 = numpy.uint64(i * random_vari)
        # if x1 > (x1 ^ input_diff):
        #     continue
        c1 = enc(x1, rounds, word_size, keys, perm_list)
        np1 = dec(c1, rounds, word_size, keys, perm_list)
        if c1 == np1:
            res += 1
        # x2 = x1 ^ input_diff
        # c2 = enc(x2, rounds, word_size, keys, perm_list)
        #
        # c3 = c1 ^ output_diff
        # c4 = c2 ^ output_diff
        #
        # x3 = dec(c3, rounds, word_size, keys, perm_list)
        # x4 = dec(c4, rounds, word_size, keys, perm_list)
        #
        # if x3 ^ x4 == input_diff:
        #     cuda.atomic.add(result_collector, 0, 1)
        #     res += 1
    result_collector[thread_index] = res


# GPU Tasks
def test():
    input_dff = INPUT_LEFT_DIFF << 32 | INPUT_RIGHT_DIFF
    output_diff = OUTPUT_LEFT_DIFF << 32 | OUTPUT_RIGHT_DIFF

    # GPU Setting
    thread_in_per_block = 2 ** THREADS_IN_PER_BLOCK_EXPO
    block_in_per_grid = 2 ** BLOCK_IN_PER_GRID_EXPO

    total_threads = thread_in_per_block * block_in_per_grid

    result = numpy.zeros((total_threads,), dtype=numpy.int32)
    key = random.randint(0, 2 ** 128)
    sub_keys = key_schedule(key)
    perm_list = [7, 4, 1, 6, 3, 0, 5, 2]

    cuda_sub_keys = cuda.to_device(sub_keys)
    cuda_result = cuda.to_device(result)
    cuda_perm_list = cuda.to_device(perm_list)
    rng_states = create_xoroshiro128p_states(total_threads, seed=1)

    start_time = time.time()

    print("Task star")
    (start_gpu_task[block_in_per_grid, thread_in_per_block](cuda_sub_keys, input_dff, output_diff, ROUNDS,
                                                            cuda_result,
                                                            WORD_SIZE, cuda_perm_list,
                                                            TASK_NUM_IN_PER_THREAD_EXPO, rng_states))
    print("Task End")
    res = 0
    for r in cuda_result:
        res += r
    print(res)
    if res == 0:
        tip = "Invalid"
    else:
        tip = math.log2(res / 2 ** (BLOCK_IN_PER_GRID_EXPO + THREADS_IN_PER_BLOCK_EXPO + TASK_NUM_IN_PER_THREAD_EXPO))
    print("w:{}".format(tip))
    print(res)
    print("Task done, time:{}".format(time.time() - start_time))


test()
