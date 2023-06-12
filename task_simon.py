import simon


def checks(x1s, key, input_diff, output_diff, rounds, offset, boomerang=True, word_size=32):
    x1s = list(x1s)
    count = 0
    if boomerang:
        for x1 in x1s:
            cipher = simon.SimonCipher(
                block_size=32, key_size=64, key=key, rounds=rounds, offset=offset
            )
            x2 = x1 ^ input_diff
            c1 = cipher.encrypt(x1)
            c2 = cipher.encrypt(x2)
            c3 = c1 ^ output_diff
            c4 = c2 ^ output_diff
            x3 = cipher.decrypt(c3)
            x4 = cipher.decrypt(c4)
            if x3 ^ x4 == input_diff:
                count += 1
    else:
        for x1 in x1s:
            cipher = simon.SimonCipher(
                block_size=32, key_size=64, key=key, rounds=rounds, offset=offset
            )
            x2 = x1 ^ input_diff
            c1 = cipher.encrypt(x1)
            c2 = cipher.encrypt(x2)
            if c1 ^ c2 == output_diff:
                count += 1
    return count

