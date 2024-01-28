import math

import numpy
from numba import cuda
import random
import time
import operator

INPUT_LEFT_DIFF = 0x00000084
INPUT_RIGHT_DIFF = 0x04008046
OUTPUT_LEFT_DIFF = 0x00000080
OUTPUT_RIGHT_DIFF = 0x00000084
WORD_SIZE = 64
THREADS_IN_PER_BLOCK_EXPO = 10
BLOCK_IN_PER_GRID_EXPO = 10
TASK_NUM_IN_PER_THREAD_EXPO = 32 - THREADS_IN_PER_BLOCK_EXPO - BLOCK_IN_PER_GRID_EXPO
ROUNDS = 1


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


@cuda.jit
def g0(block_size, temp_list):
    num = temp_list[3]
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

    temp_list[3] = operator.xor(
        operator.xor(
            operator.xor(
                operator.lshift(y3, group_size * 3),
                operator.lshift(y2, group_size * 2),
            ),
            operator.lshift(y1, group_size)
        ), y0
    )


@cuda.jit
def g1(block_size, temp_list):
    num = temp_list[4]
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
    temp_list[4] = operator.xor(
        operator.xor(
            operator.xor(
                operator.lshift(y3, group_size * 3),
                operator.lshift(y2, group_size * 2),
            ),
            operator.lshift(y1, group_size)
        ), y0
    )


@cuda.jit
def rotation(rot_size, block_size, temp_list):
    num = temp_list[2]
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

        temp_list[2] = operator.xor(
            operator.xor(
                operator.xor(
                    operator.lshift(n3, group_size * 3),
                    operator.lshift(n2, group_size * 2),
                ),
                operator.lshift(n1, group_size)
            ), n0
        )


@cuda.jit
def perm(block_size, temp_list, perm_list):
    num = temp_list[5]
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

    temp_list[5] = operator.xor(
        operator.xor(
            operator.xor(
                operator.lshift(res3, group_size * 3),
                operator.lshift(res2, group_size * 2),
            ),
            operator.lshift(res1, group_size)
        ), res0
    )


@cuda.jit
def enc(rounds, word_size, keys, temp_list, perm_list, temp_list64, thread_index):
    plaintext = temp_list64[thread_index]
    alpha = 0
    beta = 1
    block_size = word_size // 2
    p_l = operator.rshift(plaintext, block_size)
    p_r = operator.and_(plaintext, 0xFFFFFFFF)
    # init_input_gpu(p_l, block_size, temp_list)
    init_l = numpy.uint32(p_l)
    # init_input_gpu(p_r, block_size, temp_list)
    init_r = numpy.uint32(p_r)
    for i in range(rounds):
        ori_l = init_l

        temp_list[2] = init_l
        rotation(alpha, block_size, temp_list)
        rot_g0 = temp_list[2]

        temp_list[3] = rot_g0
        g0(block_size, temp_list)
        g0_out = temp_list[3]

        temp_list[2] = init_l
        rotation(beta, block_size, temp_list)
        rot_g1 = temp_list[2]

        temp_list[4] = rot_g1
        g1(block_size, temp_list)
        g1_out = temp_list[4]

        temp_list[5] = operator.xor(g0_out, g1_out)
        perm(block_size, temp_list, perm_list)
        perm_out = temp_list[5]

        init_l = operator.xor(perm_out, init_r)  # ^ keys[i]
        init_r = ori_l

    temp_list64[thread_index] = operator.xor(operator.lshift(init_l, block_size), init_r)


@cuda.jit
def start_gpu_task(keys, input_diff, output_diff, rounds, result_collector, temp_list, word_size, perm_list,
                   temp_list64, task_num):
    thread_index = operator.add(operator.mul(cuda.blockIdx.x, cuda.blockDim.x), cuda.threadIdx.x)
    res = 0
    used_list = temp_list[thread_index]
    start = thread_index * (2 ** task_num)
    end = (thread_index + 1) * (2 ** task_num)
    for i in range(start, end):
        x1 = i
        # used_list[2] = x1
        # rotation(16, 32, used_list)
        # x1 = used_list[2]
        # if x1 > (x1 ^ input_diff):
        #     continue
        temp_list64[thread_index] = x1
        enc(rounds, word_size, keys, used_list, perm_list, temp_list64, thread_index)
        c1 = temp_list64[thread_index]

        x2 = x1 ^ input_diff
        temp_list64[thread_index] = x2
        enc(rounds, word_size, keys, used_list, perm_list, temp_list64, thread_index)
        c2 = temp_list64[thread_index]

        if c1 ^ c2 == output_diff:
            res = operator.add(res, 1)

        # c3 = c1 ^ output_diff
        # c4 = c2 ^ output_diff
        #
        # dec(c3, keys, ir, offset, rounds, used_list)
        # x3 = used_list[0]
        #
        # dec(c4, keys, ir, offset, rounds, used_list)
        # x4 = used_list[0]
        # if x3 ^ x4 == input_diff:
        #     res += 2
    cuda.atomic.add(result_collector, 0, res)


# GPU Tasks
def test():
    input_dff = INPUT_LEFT_DIFF << 32 | INPUT_RIGHT_DIFF
    output_diff = OUTPUT_LEFT_DIFF << 32 | OUTPUT_RIGHT_DIFF

    # GPU Setting
    thread_in_per_block = numpy.uint64(math.pow(2, THREADS_IN_PER_BLOCK_EXPO))
    block_in_per_grid = numpy.uint64(math.pow(2, BLOCK_IN_PER_GRID_EXPO))

    total_threads = thread_in_per_block * block_in_per_grid

    result = numpy.zeros((1,), dtype=numpy.float64)
    temp_list = numpy.array([[0 for _ in range(7)] for _ in range(total_threads)], dtype=numpy.uint32)
    temp_list64 = numpy.zeros(total_threads, dtype=numpy.uint64)
    key = random.randint(0, 2 ** 128)
    sub_keys = key_schedule(key)
    perm_list = [7, 4, 1, 6, 3, 0, 5, 2]

    cuda_sub_keys = cuda.to_device(sub_keys)
    cuda_result = cuda.to_device(result)
    cuda_temp_list = cuda.to_device(temp_list)
    cuda_perm_list = cuda.to_device(perm_list)
    cuda_temp_list64 = cuda.to_device(temp_list64)
    start_time = time.time()

    print("Task star")
    (start_gpu_task[block_in_per_grid, thread_in_per_block](cuda_sub_keys, input_dff, output_diff, ROUNDS,
                                                            cuda_result,
                                                            cuda_temp_list, WORD_SIZE, cuda_perm_list,
                                                            cuda_temp_list64, TASK_NUM_IN_PER_THREAD_EXPO))
    print("Task End")
    res = cuda_result[0]
    if res == 0:
        tip = "Invalid"
    else:
        tip = math.log2(res / 2 ** (BLOCK_IN_PER_GRID_EXPO + THREADS_IN_PER_BLOCK_EXPO + TASK_NUM_IN_PER_THREAD_EXPO))
    print("w:{}".format(tip))
    print(res)
    print("Task done, time:{}".format(time.time() - start_time))


test()

# n = 0x12345678
# temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# temp[2] = n
# temp[3] = n
# temp[4] = n
# temp[5] = n
# g0(32, temp)
# g1(32, temp)
# rotation(1, 32, temp)
# perm(32, temp, [7, 4, 1, 6, 3, 0, 5, 2])
# print(temp[2])
# print(temp[3])
# print(temp[4])
# print(temp[5])
