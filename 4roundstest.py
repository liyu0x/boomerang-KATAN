import random

import katan_self


input_diff = 0x04000000
out_put = 0x20000000

key = random.randint(0,2**80)

_in1 = random.randint(0,2**32)

_in2 = _in1 ^ input_diff

ci1 = katan_self.enc32(_in1, key, 4)
ci2 = katan_self.enc32(_in2, key, 4)

ci3 = ci1 ^ out_put
ci4 = ci2 ^ out_put

_in3 = katan_self.dec32(ci3, key, 4)
_in4 = katan_self.dec32(ci4, key, 4)

print(_in3 ^ _in4 == input_diff)