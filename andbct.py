import random, katan, numpy, util

AX_BOX_INPUT_SIZE, AX_BOX_OUTPUT_SIZE = 4, 4

SMALL_REGISTER_OFFSET = 19
BIG_REGISTER_OFFSET = 0

BCT = numpy.zeros([2 ** AX_BOX_INPUT_SIZE, 2 ** AX_BOX_OUTPUT_SIZE], dtype=int)

FOUR_X_INDEXES = [SMALL_REGISTER_OFFSET + x for x in [8, 5, 3]]
FOUR_Y_INDEXES = [BIG_REGISTER_OFFSET + x for x in [12, 10, 8, 3]]


def create_bct():
    for delta_in in range(2 ** AX_BOX_INPUT_SIZE):
        for delta_out in range(2 ** AX_BOX_OUTPUT_SIZE):
            for x in range(2 ** AX_BOX_INPUT_SIZE):
                x_delta_in = x ^ delta_in
                x_delta_out = x ^ delta_out
                x_delta_in_out = x ^ delta_in ^ delta_out
                r_x = util.ax_box(x)
                r_x_delta_in = util.ax_box(x_delta_in)
                r_x_delta_out = util.ax_box(x_delta_out)
                r_x_delta_in_out = util.ax_box(x_delta_in_out)
                if r_x ^ r_x_delta_in ^ r_x_delta_out ^ r_x_delta_in_out == 0:
                    BCT[delta_in][delta_out] += 1


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
    total = 2 ** 10
    active = 0
    delta_left_in = 5
    delta_left_out = 9
    # delta_right_out = 1

    delta_right_in = 1
    for _ in range(total):
        p1 = random.randint(0, 2 ** 32)
        p2 = compute_different(p1, delta_left_in, [3, 8, 10, 12])
        p2 = compute_different(p2, delta_right_in, [18])
        c1 = katan.enc32(p1, 0xDDFC)
        c2 = katan.enc32(p2, 0xDDFC)
        c3 = compute_different(c1, delta_left_out, [2, 7, 9, 11])
        c4 = compute_different(c2, delta_left_out, [2, 7, 9, 11])
        p3 = katan.dec32(c3, 0xDDFC)
        p4 = katan.dec32(c4, 0xDDFC)
        if get_difference(p3, p4, [18]) == delta_right_in:
            active += 1
    print(active)


if __name__ == "__main__":
    create_bct()
    check_validation_bct()
