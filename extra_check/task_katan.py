import katan_cipher
import math
import random


def checks(
    cipher: katan_cipher.KATAN,
    x1s,
    input_diff,
    input_switch_diff,
    output_switch_diff,
    output_diff,
    rounds,
    switch_rounds,
):
    count = 0
    switch_start_round = int(rounds / 2) - int(switch_rounds / 2)
    switch_end_round = switch_start_round + switch_rounds
    x1s = list(x1s)

    e0_en_count = 0
    em_en_count = 0
    e1_en_count = 0

    em_de_count = 0
    e1_de_count = 0

    for x1 in x1s:
        x2 = x1 ^ input_diff
        c1 = cipher.enc(plaintext=x1, from_round=0, to_round=rounds)
        c2 = cipher.enc(plaintext=x2, from_round=0, to_round=rounds)
        c3 = c1 ^ output_diff
        c4 = c2 ^ output_diff

        m2 = cipher.dec(ciphertext=c2, from_round=rounds, to_round=switch_end_round)
        m4 = cipher.dec(ciphertext=c4, from_round=rounds, to_round=switch_end_round)
        m1 = cipher.dec(ciphertext=c1, from_round=rounds, to_round=switch_end_round)
        m3 = cipher.dec(ciphertext=c3, from_round=rounds, to_round=switch_end_round)

        if m2 ^ m4 == out_switch_diff and m1 ^ m3 == out_switch_diff:
            m3 = cipher.dec(ciphertext=c3, from_round=rounds, to_round=switch_start_round)
            m4 = cipher.dec(ciphertext=c4, from_round=rounds, to_round=switch_start_round)
            if m3 ^ m4 == input_switch_diff:
                em_de_count += 1
            
        

        m3 = cipher.dec(ciphertext=c3, from_round=rounds, to_round=switch_start_round)
        m4 = cipher.dec(ciphertext=c4, from_round=rounds, to_round=switch_start_round)
        if m3 ^ m4 == input_switch_diff:
            em_en_count += 1

        p3 = cipher.dec(ciphertext=c3, from_round=rounds, to_round=0)
        p4 = cipher.dec(ciphertext=c4, from_round=rounds, to_round=0)
        if p3 ^ p4 == in_diff:
            count += 1

    print("the number of decryption em: {}".format(em_de_count))
    print("the number of encryption em: {}".format(em_en_count))
    print("the number of decryption e1: {}".format(e1_de_count))
    return count


if __name__ == "__main__":
    in_diff = 0x00000800
    in_switch_diff = 0x04000008
    out_switch_diff = 0x00000800
    out_diff = 0x08000005
    test_n = 2**10
    cipher = katan_cipher.KATAN(master_key=0xFF1243)
    records = set()
    count = 0
    result = 0
    task_list = []
    while count < test_n:
        x1 = random.randint(0, 2**32)
        if x1 in records:
            continue
        if x1 > (x1 ^ in_diff):
            continue
        count += 1
        records.add(x1)
    records = list(records)
    res = checks(cipher, records, in_diff, in_switch_diff, out_switch_diff, out_diff, 34, 4)
    prob = res/test_n
    weight = math.log2(prob)
    print("weight:{}".format(weight))
