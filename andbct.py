import random, katan, numpy, util

BIG_BCT_INPUT_SIZE, BIG_BCT_OUTPUT_SIZE = 4, 4
SMALL_BCT_INPUT_SIZE, SMALL_BCT_OUTPUT_SIZE = 2, 2

SMALL_REGISTER_OFFSET = 19
BIG_REGISTER_OFFSET = 0

B_BCT = numpy.zeros([2 ** BIG_BCT_INPUT_SIZE, 2 ** BIG_BCT_OUTPUT_SIZE], dtype=int)
S_BCT = numpy.zeros([2 ** SMALL_BCT_INPUT_SIZE, 2 ** SMALL_BCT_OUTPUT_SIZE], dtype=int)

FOUR_X_INDEXES = [SMALL_REGISTER_OFFSET + x for x in [8, 5, 3]]
FOUR_Y_INDEXES = [BIG_REGISTER_OFFSET + x for x in [12, 10, 8, 3]]


def create_bct():
    for delta_in in range(2 ** BIG_BCT_INPUT_SIZE):
        for delta_out in range(2 ** BIG_BCT_OUTPUT_SIZE):
            for x in range(2 ** BIG_BCT_INPUT_SIZE):
                x_delta_in = x ^ delta_in
                x_delta_out = x ^ delta_out
                x_delta_in_out = x ^ delta_in ^ delta_out
                r_x = util.ax_box(x)
                r_x_delta_in = util.ax_box(x_delta_in)
                r_x_delta_out = util.ax_box(x_delta_out)
                r_x_delta_in_out = util.ax_box(x_delta_in_out)
                if r_x ^ r_x_delta_in ^ r_x_delta_out ^ r_x_delta_in_out == 0:
                    B_BCT[delta_in][delta_out] += 1
    for delta_in in range(2 ** SMALL_BCT_INPUT_SIZE):
        for delta_out in range(2 ** SMALL_BCT_OUTPUT_SIZE):
            for x in range(2 ** SMALL_BCT_INPUT_SIZE):
                x_delta_in = x ^ delta_in
                x_delta_out = x ^ delta_out
                x_delta_in_out = x ^ delta_in ^ delta_out
                r_x = util.ax_box_2_bits(x)
                r_x_delta_in = util.ax_box_2_bits(x_delta_in)
                r_x_delta_out = util.ax_box_2_bits(x_delta_out)
                r_x_delta_in_out = util.ax_box_2_bits(x_delta_in_out)
                if r_x ^ r_x_delta_in ^ r_x_delta_out ^ r_x_delta_in_out == 0:
                    S_BCT[delta_in][delta_out] += 1


def compute_different(x, delta, bits_indexes):
    x_bits = util.num_to_bits(x)

    for ind in bits_indexes:
        x_bits[ind] ^= (delta & 0x1)
        delta >>= 1
    return util.bits_to_num(x_bits)


def get_difference(x1, x2, bits_indexes):
    delta = x1 ^ x2
    delta_bits = util.num_to_bits(delta)
    n_res = []
    for ind in bits_indexes:
        n_res.append(delta_bits[ind])
    return util.bits_to_num(n_res)


def check_validation_bct():
    active = 0
    big_register_delta_and_in = 7
    big_register_delta_and_out = 13
    small_register_delta_and_in = 2
    small_register_delta_and_out = 3
    left_diff = 5
    right_diff = 1
    total = 2 ** 10
    key = 0xFFFFFFFFFFFFFFFFFFFF
    for _ in range(total):
        p1 = random.randint(1, 2 ** 32)

        p2 = p1
        p2 = compute_different(p1, big_register_delta_and_in, [3, 8, 10, 12])
        p2 = compute_different(p2, small_register_delta_and_in, [19 + 5, 19 + 8])

        p2 = compute_different(p2, right_diff, [18])
        p2 = compute_different(p2, left_diff, [19 + 3, 19 + 7, 19 + 12])

        c1 = katan.enc32(p1, key)
        c2 = katan.enc32(p2, key)

        c3 = c1
        c3 = compute_different(c3, big_register_delta_and_out, [3 + 1, 8 + 1, 10 + 1, 12 + 1])
        c3 = compute_different(c3, small_register_delta_and_out, [19 + 5 + 1, 19 + 8 + 1])

        c4 = c2
        c4 = compute_different(c4, big_register_delta_and_out, [3 + 1, 8 + 1, 10 + 1, 12 + 1])
        c4 = compute_different(c4, small_register_delta_and_out, [19 + 5 + 1, 19 + 8 + 1])

        p3 = katan.dec32(c3, key)
        p4 = katan.dec32(c4, key)
        r1 = get_difference(p3, p4, [18])
        l1 = get_difference(p3, p4, [19 + 3, 19 + 7, 19 + 12])
        if l1 == left_diff and r1 == right_diff:
            active += 1
    print(active)


if __name__ == "__main__":
    create_bct()
    check_validation_bct()
    print("done")
