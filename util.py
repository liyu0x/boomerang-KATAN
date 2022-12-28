def num_to_bits(num: int, length=32):
    bits = []
    for i in range(length):
        bits.append(num & 0x1)
        num >>= 1
    return bits


def bits_to_num(bits: list):
    res = 0
    for i, b in enumerate(bits):
        res += (b << i)
    return res


def ax_box(x, bit_size):
    res = 0
    for i in range(bit_size - 1, 0, -2):
        x0 = x >> i & 0x1
        x1 = x >> (i - 1) & 0x1
        res ^= (x0 & x1)
    return res


def ax_box2(x):
    res = 0
    x0 = x >> 2 & 0x11
    x1 = x & 0x11
    res ^= (x0 & x1)
    return res


def ax_box_2_bits(x):
    x0 = x >> 1 & 0x1
    x1 = x & 0x1
    return x0 & x1
