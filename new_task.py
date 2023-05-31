import katan

INPUD_DIFF = 0x10040080
OUTPUT_DIFF = 0x00300420


ROUNDS = 83

def check(x1, key):
    x2 = x1 ^ INPUD_DIFF
    c1 = katan.enc32(x1, key, ROUNDS)
    c2 = katan.enc32(x2, key, ROUNDS)
    c3 = c1 ^ OUTPUT_DIFF
    c4 = c2 ^ OUTPUT_DIFF
    x3 = katan.dec32(c3, key, ROUNDS)
    x4 = katan.dec32(c4, key, ROUNDS)
    if x3 ^ x4 == INPUD_DIFF:
        return 1
    return 0