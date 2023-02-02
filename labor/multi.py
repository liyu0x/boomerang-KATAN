import sys

sys.path.append('../')

import util


def extract_diff(differ: int, positions: list):
    bits = util.num_to_bits(differ)
    res = [bits[i] for i in positions]
    return util.bits_to_num(res)


delta_in = 0x03980312
nabla_out = 0x4720C208

di = extract_diff(delta_in, [5, 8])
do = extract_diff(nabla_out, [6, 9])

print("in:{0},out:{1}".format(di, do))
