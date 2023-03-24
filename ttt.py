n = 2
m = 1

# Define the set of input differences
input_diffs = [1 << i for i in range(n)]

# Initialize the DDT
ddt = [[0 for j in range(2**m)] for i in range(2**n)]

# Compute the DDT
for in_diff1 in input_diffs:
    for in_diff2 in input_diffs:
        out_diff = 0
        for i in range(m):
            if (in_diff1 >> i) & (in_diff2 >> i) & 1:
                out_diff |= 1 << i
        in_diff1_idx = in_diff1.bit_length() - 1
        out_diff_idx = out_diff.bit_length() - 1
        ddt[in_diff1_idx][out_diff_idx] += 1

# Print the DDT
for row in ddt:
    print(row)
