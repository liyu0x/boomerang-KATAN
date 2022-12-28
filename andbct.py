import random, katan, numpy, util

BCT_INPUT_SIZE, BCT_OUTPUT_SIZE = 2, 2
N_BCT_INPUT_SIZE, N_BCT_OUTPUT_SIZE = 6, 6
SMALL_REGISTER_OFFSET = 19
BIG_REGISTER_OFFSET = 0

BCT = numpy.zeros([2 ** BCT_INPUT_SIZE, 2 ** BCT_OUTPUT_SIZE], dtype=int)
BCT2 = numpy.zeros([2 ** N_BCT_INPUT_SIZE, 2 ** N_BCT_OUTPUT_SIZE], dtype=int)
DDT = numpy.zeros([2 ** 4, 2 ** 2], dtype=int)
DDT2 = numpy.zeros([2 ** 3, 2 ** 3], dtype=int)


def create_bct():
    for delta_in in range(2 ** BCT_INPUT_SIZE):
        for delta_out in range(2 ** BCT_OUTPUT_SIZE):
            for x in range(2 ** BCT_INPUT_SIZE):
                x_delta_in = x ^ delta_in
                x_delta_out = x ^ delta_out
                x_delta_in_out = x ^ delta_in ^ delta_out
                r_x = util.ax_box(x, BCT_INPUT_SIZE)
                r_x_delta_in = util.ax_box(x_delta_in, BCT_INPUT_SIZE)
                r_x_delta_out = util.ax_box(x_delta_out, BCT_INPUT_SIZE)
                r_x_delta_in_out = util.ax_box(x_delta_in_out, BCT_INPUT_SIZE)
                if r_x ^ r_x_delta_in ^ r_x_delta_out ^ r_x_delta_in_out == 0:
                    BCT[delta_in][delta_out] += 1


def create_bct2():
    for delta_in in range(2 ** N_BCT_INPUT_SIZE):
        for delta_out in range(2 ** N_BCT_OUTPUT_SIZE):
            for x in range(2 ** N_BCT_INPUT_SIZE):
                x_delta_in = x ^ delta_in
                x_delta_out = x ^ delta_out
                x_delta_in_out = x ^ delta_in ^ delta_out
                r_x = util.ax_box2(x)
                r_x_delta_in = util.ax_box2(x_delta_in)
                r_x_delta_out = util.ax_box2(x_delta_out)
                r_x_delta_in_out = util.ax_box2(x_delta_in_out)
                if r_x ^ r_x_delta_in ^ r_x_delta_out ^ r_x_delta_in_out == 0:
                    BCT2[delta_in][delta_out] += 1


def create_ddt():
    for delta_in in range(2 ** 4):
        for delta_out in range(2 ** 2):
            for x0 in range(2 ** 4):
                x1 = x0 ^ delta_in

                x0_0 = x0 & 0x11
                x0_1 = (x0 >> 2) & 0x11
                y0 = x0_0 & x0_1

                x1_0 = x1 & 0x11
                x1_1 = (x1 >> 2) & 0x11
                y1 = x1_0 & x1_1

                out = y0 ^ y1

                if out == delta_out:
                    DDT[delta_in][delta_out] += 1


def create_ddt2():
    for delta_in in range(2 ** 3):
        for delta_out in range(2 ** 3):
            for x0 in range(2 ** 3):
                x1 = x0 ^ delta_in

                x0_0 = x0 & 0x1
                x0_1 = (x0 >> 1) & 0x1
                x0_2 = (x0 >> 2) & 0x1
                y0 = x0_0 ^ (x0_1 & x0_2)

                x1_0 = x1 & 0x1
                x1_1 = (x1 >> 1) & 0x1
                x1_2 = (x1 >> 2) & 0x1
                y1 = x1_0 ^ (x1_1 & x1_2)

                y0 = x0 & 0x0110 | y0
                y1 = x1 & 0x0110 | y1

                out = y0 ^ y1

                if out == delta_out:
                    DDT2[delta_in][delta_out] += 1


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
    big_register_delta_and_in = 12
    big_register_delta_and_out = 12
    small_register_delta_and_in = 2
    small_register_delta_and_out = 1
    left_diff = 0
    right_diff = 1
    total = 2 ** 10
    key = 0xFFFFFFFFFFFFFFFFFFFF
    for _ in range(total):
        p1 = random.randint(1, 2 ** 32)

        p2 = p1
        p2 = compute_different(p2, big_register_delta_and_in, [3, 8, 10, 12])
        p2 = compute_different(p2, small_register_delta_and_in, [19 + 5, 19 + 8])

        p2 = compute_different(p2, right_diff, [18])
        p2 = compute_different(p2, left_diff, [19 + 12])

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
        l1 = get_difference(p3, p4, [19 + 12])
        if l1 == left_diff and r1 == right_diff:
            active += 1
    print(active)


if __name__ == "__main__":
    create_bct()
    # check_validation_bct()
    create_ddt()
    create_bct2()
    print("done")
