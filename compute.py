import math

ws = {24:4, 25:524, 26:34056}


prob = 0
for w in ws:
    prob += math.pow(2, -w) * ws[w]
prob *= prob
print(math.log2(prob))