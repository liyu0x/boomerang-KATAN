import numpy
import numpy as np

g0_s_box = {0x0: 0x0, 0x1: 0x1, 0x2: 0x2, 0x3: 0xB, 0x4: 0x4, 0x5: 0x5, 0x6: 0x6, 0x7: 0xF, 0x8: 0x8,
            0x9: 0x9,
            0xA: 0xA, 0xB: 0x3, 0xC: 0xD, 0xD: 0xC, 0xE: 0x7, 0xF: 0xE}

g1_s_box = {0x0: 0x0, 0x1: 0x1, 0x2: 0x2, 0x3: 0x3, 0x4: 0x4, 0x5: 0x7, 0x6: 0x6, 0x7: 0x5, 0x8: 0x8,
            0x9: 0x9,
            0xA: 0xE, 0xB: 0xD, 0xC: 0xC, 0xD: 0xF, 0xE: 0xA, 0xF: 0xB}


def create_ddt(s_box):
    ddt = np.zeros((2 ** 4, 2 ** 4), dtype=numpy.int8)
    for x1 in range(2 ** 4):
        for x2 in range(2 ** 4):
            input_diff = x1 ^ x2
            y1 = s_box[x1]
            y2 = s_box[x2]
            output_diff = y1 ^ y2
            ddt[input_diff][output_diff] += 1
    return ddt


def create_bct():
    bct = np.zeros((2 ** 4, 2 ** 4), dtype=numpy.int8)
    for x1 in range(2 ** 4):
        for delta_in in range(2 ** 4):
            for delta_out in range(2 ** 4):
                y = g0_s_box[x1] ^ g1_s_box[x1] ^ g0_s_box[x1 ^ delta_in] ^ g1_s_box[x1 ^ delta_in] ^ g0_s_box[
                    x1 ^ delta_out] ^ g1_s_box[x1 ^ delta_out] ^ g0_s_box[x1 ^ delta_in ^ delta_out] ^ g1_s_box[
                        x1 ^ delta_in ^ delta_out]
                if y == 0:
                    bct[delta_in][delta_out] += 1
    return bct

    # g0_ddt = create_ddt(g0_s_box)


# g1_ddt = create_ddt(g1_s_box)
BCT = create_bct()
# np.savetxt('data1.csv', g1_ddt, delimiter=',')
np.savetxt('data2.csv', BCT, delimiter=',')
