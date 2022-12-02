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


def ax_box(x):
    x0 = x >> 3 & 0x1
    x1 = x >> 2 & 0x1
    x2 = x >> 1 & 0x1
    x3 = x & 0x1
    return (x0 & x1) ^ (x2 & x3)
