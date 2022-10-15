IR = 0xb0103459eb3d0711345ca5bedf62a7295bec3354fd47cc50f0918293b3d57f8

KATAN_32_X = [12, 7, 8, 5, 3]
KATAN_48_X = [18, 12, 15, 7, 6]
KATAN_64_X = [24, 15, 20, 11, 9]
KATAN_X = []

KATAN_32_Y = [18, 7, 12, 10, 8, 3]
KATAN_48_Y = [28, 19, 21, 13, 15, 6]
KATAN_64_Y = [38, 25, 33, 21, 14, 9]
KATAN_Y = []

SUB_KEYS = list()
L1, L2 = [], []


def num_to_bits(num: int, length=32):
    binary = bin(num)[2:]
    res = [int(c) for c in binary]
    for z in range(len(res), length):
        res.insert(0, 0)
    return res


def bits_to_num(nums: list):
    res = ''
    for n in nums:
        res += str(n)
    return int(res, 2)


def init_register(length_1: int, length_2: int):
    global L1, L2
    L1, L2 = [0 for i in range(length_1)], [0 for i in range(length_2)]


def into_registers(bits: list):
    i = 0
    for _ in range(len(L1)):
        L1[i] = bits[i]
        i += 1
    j = 0
    for _ in range(i, len(bits)):
        L2[j] = bits[i]
        j += 1
        i += 1


def init_sub_key(key: int, num: int):
    global SUB_KEYS
    SUB_KEYS = num_to_bits(key, 80)
    for i in range(80, num):
        sub_key = SUB_KEYS[i - 80] ^ SUB_KEYS[i - 61] ^ SUB_KEYS[i - 50] ^ SUB_KEYS[i - 13]
        SUB_KEYS.append(sub_key)


def f_a(k_a):
    ir = IR & 0x3FF  # b1111111111
    return L1[KATAN_X[0]] ^ L1[KATAN_X[1]] ^ (L1[KATAN_X[2]] & L1[KATAN_X[3]]) ^ (L1[KATAN_X[5]] & ir) ^ k_a


def f_b(k_b):
    return L2[KATAN_Y[0]] ^ L2[KATAN_Y[1]] ^ (L2[KATAN_Y[2]] & L2[KATAN_Y[3]]) ^ (L2[KATAN_Y[4]] & L2[KATAN_Y[5]]) ^ k_b


def round_func(round_num: int):
    global L1, L2, IR
    i = 0
    while True:
        if i == round_num:
            break
        a = f_a(SUB_KEYS[i * 2])
        b = f_b(SUB_KEYS[i * 2 + 1])
        L1[0] = b
        L2[0] = a
        i += 1
        if i % 10 == 0:
            IR >>= 10


def katan32(plaintext: int, key: int):
    global KATAN_X, KATAN_Y
    length = 32
    KATAN_X = KATAN_32_X
    KATAN_Y = KATAN_32_Y
    init_register(13, 19)
    bits = num_to_bits(plaintext, length)
    into_registers(bits)
    init_sub_key(key, 253)
    round_func(4)


if __name__ == "__main__":
    p = 0xFFFFFFFFFFFFFFFF
    k = 0xFFFFFFFFFFFFFFFFFFFF
    katan32(p, k)
