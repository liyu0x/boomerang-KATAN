import numpy as np

S_BOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x00, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
S_BOX_REV = [S_BOX.index(i) for i in range(len(S_BOX))]
P_BOX = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22,
         38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44,
         60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]
P_BOX_REV = [P_BOX.index(i) for i in range(len(P_BOX))]


def apply_s_rev_box(num: int):
    result = 0
    for i in range(16):
        result |= S_BOX_REV[(num >> (i * 4)) & 0xf] << (4 * i)
    return result


def apply_s_box(num: int):
    result = 0
    for i in range(16):
        result |= S_BOX[(num >> (4 * i)) & 0xf] << (4 * i)
    return result


def apply_p_box(num: int):
    result = 0
    for i in range(64):
        result |= ((num >> i) & 0x1) << P_BOX[i]
    return result


def apply_p_rev_box(num: int):
    result = 0
    for i in range(64):
        result |= ((num >> i) & 0x1) << P_BOX_REV[i]
    return result


def encrypt(p: int, key: int):
    p ^= key
    result = apply_s_box(p)
    return apply_p_box(result)


def decrypt(c: int, key: int):
    c = apply_p_rev_box(c)
    result = apply_s_rev_box(c)
    result ^= key
    return result


def diff_decrypt(c: int):
    c = apply_p_rev_box(c)
    result = apply_s_rev_box(c)
    return result


BCT = np.zeros((16, 16), dtype=int)

for i in range(16):
    for j in range(16):
        for n in range(16):
            out = S_BOX_REV[S_BOX[n] ^ j] ^ S_BOX_REV[
                S_BOX[n ^ i] ^ j]
            if out == i:
                BCT[i][j] += 1

print(BCT)