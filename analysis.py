def compute_difference(x, y, dx, dy):
    return (x & y) ^ ((x ^ dx) & (y ^ dy))

