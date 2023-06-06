import katan_cipher


def checks(x1s, key, input_diff, output_diff, rounds, offset, boomerang=True, version=32):
    x1s = list(x1s)
    count = 0
    cipher = katan_cipher.KATAN(master_key=key, version=version)
    if boomerang:
        for x1 in x1s:
            x2 = x1 ^ input_diff
            c1 = cipher.enc(plaintext=x1, from_round=offset, to_round=rounds)
            c2 = cipher.enc(plaintext=x2, from_round=offset, to_round=rounds)
            c3 = c1 ^ output_diff
            c4 = c2 ^ output_diff
            x3 = cipher.dec(ciphertext=c3, from_round=rounds, to_round=offset)
            x4 = cipher.dec(ciphertext=c4, from_round=rounds, to_round=offset)
            if x3 ^ x4 == input_diff:
                count += 1
    else:
        for x1 in x1s:
            x2 = x1 ^ input_diff
            c1 = cipher.enc(plaintext=x1, from_round=offset, to_round=rounds)
            c2 = cipher.enc(plaintext=x2, from_round=offset, to_round=rounds)
            if c1 ^ c2 == output_diff:
                count += 1
    return count


if __name__ == "__main__":
    import math
    import random
    cipher = katan_cipher.KATAN(master_key=0xFF1243)
    INPUT_DIFF = 0x00002000
    OUTPUT_DIFF = 0x01082102
    CASE = 2**10
    FROM_ROUNDS = 0
    ROUNDS = 26

    count = 0
    record = {}
    i = 0
    while i < CASE:
        x1 = random.randint(0, 2**32)
        if x1 in record:
            continue
        x2 = x1 ^ INPUT_DIFF
        if x1 > x2:
            continue
        i+=1
        record[x1] = x2
        c1 = cipher.enc(plaintext=x1, from_round=FROM_ROUNDS, to_round=ROUNDS)
        c2 = cipher.enc(plaintext=x2, from_round=FROM_ROUNDS, to_round=ROUNDS)
        if c1 ^ c2 == OUTPUT_DIFF:
            count += 1

    if count != 0:
        prob = count/CASE
        print(math.log2(prob))