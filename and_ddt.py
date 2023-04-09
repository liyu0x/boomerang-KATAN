import numpy

ddt = numpy.zeros((4, 2), dtype=int)


def ax_box_2_bits(x):
    x0 = x >> 1 & 0b1
    x1 = x & 0b1
    return x0 & x1


for i in range(4):
    for delta_in in range(4):
        _in = i ^ delta_in
        y = ax_box_2_bits(i)
        _y = ax_box_2_bits(_in)
        ddt[delta_in][y ^ _y] += 1
