F = 16


def madd(x, y):
    return (x + y) % F


def mred(x, y):
    return (x - y) % F


def compute():
    a = 0
    a_ = 0
    b = 0
    b_ = 0

    for t1 in range(2 ** F):
        for t2 in range(2 ** F):
            a_ = t1
            b_ = t2
            for x in range(2 ** F):
                for x_ in range(2 ** F):
                    if abct(x, x_, a, a_, b, b_):
                        a = 1


def abct(x, x_, a, a_, b, b_):
    return mred(madd(x, x_) ^ b, (x_, b_)) ^ mred(madd((x ^ a), (x_ ^ a_)) ^ b, x_ ^ a_ ^ b_) == a
