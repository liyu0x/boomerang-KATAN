# verify boomerang-style switch
import katan
import util
import and_bct
import random

TEST_ROUND = 4

L2_BEFORE = [3, 8, 10, 12]
L2_AFTER = [i + TEST_ROUND for i in L2_BEFORE]
L1_BEFORE = [5, 8]
L1_AFTER = [i + TEST_ROUND for i in L1_BEFORE]
L1_R = [12]
L2_R = [18]


def compute_different(register: list, delta, bits_indexes):
    new_reg = register[:]
    for ind in bits_indexes:
        new_reg[ind] ^= (delta & 0b1)
        delta >>= 1
    return new_reg


def get_difference(reg1, reg2, bits_indexes):
    n_res = []
    for ind in bits_indexes:
        n_res.append(reg1[ind] ^ reg2[ind])
    return util.bits_to_num(n_res)


def verify_one_round_switch(p1, l1_in, l1_out, l2_in, l2_out, key):
    p1 = 2541990570
    l1_r_diff, l2_r_diff = random.randint(0, 1), random.randint(0, 1)
    bits = util.num_to_bits(p1)
    p1_l1, p1_l2 = bits[19:], bits[:19]

    p2_l1 = compute_different(p1_l1, l1_in, L1_BEFORE)
    p2_l1 = compute_different(p2_l1, l1_r_diff, L1_R)
    p2_l2 = compute_different(p1_l2, l2_in, L2_BEFORE)
    p2_l2 = compute_different(p2_l2, l2_r_diff, L2_R)

    c1_l1, c1_l2 = katan.enc32_bit(p1_l1, p1_l2, key, TEST_ROUND)
    c2_l1, c2_l2 = katan.enc32_bit(p2_l1, p2_l2, key, TEST_ROUND)

    c3_l1 = compute_different(c1_l1, l1_out, L1_AFTER)
    c3_l2 = compute_different(c1_l2, l2_out, L2_AFTER)

    c4_l1 = compute_different(c2_l1, l1_out, L1_AFTER)
    c4_l2 = compute_different(c2_l2, l2_out, L2_AFTER)

    p3_l1, p3_l2 = katan.dec32_bit(c3_l1, c3_l2, key, TEST_ROUND)
    p4_l1, p4_l2 = katan.dec32_bit(c4_l1, c4_l2, key, TEST_ROUND)

    diff_l1 = get_difference(p3_l1, p4_l1, L1_R)
    diff_l2 = get_difference(p3_l2, p4_l2, L2_R)

    if diff_l2 == l2_r_diff and diff_l1 == l1_r_diff:
        return True
    return False


def verify():
    l1_bct = and_bct.create_and_bct(and_bct.general_and_operation, 1)
    l2_bct = and_bct.create_and_bct(and_bct.general_and_xor_connection, 2)

    l1_in, l1_out = random.randint(0, 3), random.randint(0, 3)
    l2_in, l2_out = random.randint(0, 15), random.randint(0, 15)

    key = random.randint(0, 2 ** 32)
    test_num = 100
    counter = 0
    for i in range(test_num):
        p = random.randint(0, 2 ** 32)
        if verify_one_round_switch(p, l1_in, l1_out, l2_in, l2_out, key):
            counter += 1

    p1 = l1_bct[l1_in][l1_out]
    p2 = l2_bct[l2_in][l2_out]
    p = p1 * p2

    print("L1_IN:{0},L1_OUT:{1},L2_IN:{2},L2_OUT:{3},Counter:{4}".format(l1_in, l1_out, l2_in, l2_out, counter))
    if p == 0:
        assert counter == 0
    else:
        assert counter == test_num
    print("pass")


# verify

for i in range(10000):
    verify()
