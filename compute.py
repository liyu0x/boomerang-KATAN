import math

ws = {24: 4, 25: 524, 26: 34056, 27: 911861, 28: 665060, 29: 621393, }

prob = 0
for w in ws:
    prob += math.pow(2, -w * 2) * ws[w] * ws[w]
print(math.log2(prob))
