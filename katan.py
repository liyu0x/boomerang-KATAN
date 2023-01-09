import util

TEST_ROUNDS = 1

IR = (
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
    0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
    1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
    0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
    1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
    1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
    1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
    0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
    1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0
)
X_INDEXES_32 = [12, 7, 8, 5, 3]
X_INDEXES_48 = [18, 12, 15, 7, 6]
X_INDEXES_64 = [24, 15, 20, 11, 9]
X_INDEXES = []

Y_INDEXES_32 = [18, 7, 12, 10, 8, 3]
Y_INDEXES_48 = [28, 19, 21, 13, 15, 6]
Y_INDEXES_64 = [38, 25, 33, 21, 14, 9]
Y_INDEXES = []

SUB_KEYS = list()
L1, L2 = [], []


def init_sub_key(key: int):
    global SUB_KEYS
    SUB_KEYS = util.num_to_bits(key, 80)
    for i in range(80, 508):
        sub_key = SUB_KEYS[i - 80] ^ SUB_KEYS[i - 61] ^ SUB_KEYS[i - 50] ^ SUB_KEYS[i - 13]
        SUB_KEYS.append(sub_key)


def round_enc_func(round_num: int):
    global L1, L2
    for i in range(round_num):
        if i == round_num:
            break
        a = L1[X_INDEXES[0]] ^ L1[X_INDEXES[1]] ^ (L1[X_INDEXES[2]] & L1[X_INDEXES[3]]) ^ (L1[X_INDEXES[4]] & IR[i]) ^ \
            SUB_KEYS[
                i * 2]
        b = L2[Y_INDEXES[0]] ^ L2[Y_INDEXES[1]] ^ (L2[Y_INDEXES[2]] & L2[Y_INDEXES[3]]) ^ (
                L2[Y_INDEXES[4]] & L2[Y_INDEXES[5]]) ^ \
            SUB_KEYS[i * 2 + 1]
        L1.pop()
        L2.pop()
        L1.insert(0, b)
        L2.insert(0, a)


def round_dec_func(round_num: int):
    global L1, L2
    for i in range(round_num - 1, -1, -1):
        a = L2[0] ^ L1[X_INDEXES[1] + 1] ^ (L1[X_INDEXES[2] + 1] & L1[X_INDEXES[3] + 1]) ^ (
                L1[X_INDEXES[4] + 1] & IR[i]) ^ \
            SUB_KEYS[i * 2]
        b = L1[0] ^ L2[Y_INDEXES[1] + 1] ^ (L2[Y_INDEXES[2] + 1] & L2[Y_INDEXES[3] + 1]) ^ (
                L2[Y_INDEXES[4] + 1] & L2[Y_INDEXES[5] + 1]) ^ \
            SUB_KEYS[i * 2 + 1]
        L1.pop(0)
        L1.append(a)
        L2.pop(0)
        L2.append(b)


def enc32(plaintext: int, key: int):
    global X_INDEXES, Y_INDEXES, L1, L2
    X_INDEXES = X_INDEXES_32
    Y_INDEXES = Y_INDEXES_32
    init_sub_key(key)
    bits = util.num_to_bits(plaintext)
    L2 = bits[:19]
    L1 = bits[19:]
    round_enc_func(TEST_ROUNDS)
    return get_result()


def enc32_bit(l1, l2, key: int):
    global X_INDEXES, Y_INDEXES, L1, L2
    X_INDEXES = X_INDEXES_32
    Y_INDEXES = Y_INDEXES_32
    init_sub_key(key)
    L2 = l2
    L1 = l1
    round_enc_func(TEST_ROUNDS)
    return L1, L2


def dec32(cipher: int, key: int):
    global X_INDEXES, Y_INDEXES, L1, L2
    X_INDEXES = X_INDEXES_32
    Y_INDEXES = Y_INDEXES_32
    bits = util.num_to_bits(cipher)
    L2 = bits[:19]
    L1 = bits[19:]
    init_sub_key(key)
    round_dec_func(TEST_ROUNDS)
    return get_result()


def dec32_bit(l1, l2, key: int):
    global X_INDEXES, Y_INDEXES, L1, L2
    X_INDEXES = X_INDEXES_32
    Y_INDEXES = Y_INDEXES_32
    L2 = l2
    L1 = l1
    init_sub_key(key)
    round_dec_func(TEST_ROUNDS)
    return L1, L2


def get_result():
    c = L2 + L1
    return util.bits_to_num(c)


def verify():
    p = 0xFFFFFFFF
    c = 0x432E61DA
    key = 0x0
    cipher = enc32(p, key)
    if c != cipher:
        print("ENCRYPTION ERROR")
        return
    plaintext = dec32(cipher, key)
    if plaintext != p:
        print("DECRYPTION ERROR")
    print("verified")


if __name__ == "__main__":
    verify()
