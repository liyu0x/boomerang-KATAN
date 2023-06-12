import random
import math
import multiprocessing
#from task_katan import checks
from task_simon import checks


POOL = multiprocessing.get_context("fork").Pool(processes=12)
INPUD_DIFF = 0x0002820C
SWITCH_INPUT_DIFF = 0x20000000 
SWITCH_OUTPUT_DIFF = 0x00000200
OUTPUT_DIFF = 0x08300008
ROUNDS = 8
SWITCH_ROUNDS = 1
SWITCH_START_ROUNDS = int(ROUNDS/2)
OFFSET = 0
WORD_SIZE = 32


def varify(in_diff, out_diff, rounds, boomerang, offset=0):
    test_n = 2**20
    key = random.randint(0, 2**WORD_SIZE)
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
    batch_size = 1000
    batch_num = int(len(records) / batch_size)
    for i in range(0, batch_num):
        task_list.append(
            POOL.apply_async(
                checks,
                args=(
                    records[i * batch_size : i * batch_size + batch_size],
                    key,
                    in_diff,
                    out_diff,
                    rounds,
                    offset,
                    boomerang,
                    WORD_SIZE
                ),
            )
        )
    for task in task_list:
        result += task.get()
    if result == 0:
        return "Invaild"
    prob = result / (batch_size* batch_num)
    return str(math.log2(prob))


if __name__ == "__main__":
    # check boomerang distinguisher prob
    res = varify(INPUD_DIFF, OUTPUT_DIFF, ROUNDS, True)
    print("boomerang distinguisher: {}\n".format(res))

    # check upper tail
    # res = varify(INPUD_DIFF, SWITCH_INPUT_DIFF, SWITCH_START_ROUNDS-1, False)
    # print("boomerang upper trail: {}\n".format(res))

    # # check lower trail
    # res = varify(SWITCH_OUTPUT_DIFF, OUTPUT_DIFF, ROUNDS-1, False, SWITCH_START_ROUNDS+1)
    # print("boomerang lower trail: {}\n".format(res))

    # # check upper boomerang
    # res = varify(INPUD_DIFF, SWITCH_OUTPUT_DIFF, SWITCH_START_ROUNDS+1, True)
    # print("upper boomerang distinguisher: {}\n".format(res))

    # # check lower boomerang
    # res = varify(SWITCH_INPUT_DIFF, OUTPUT_DIFF, ROUNDS-SWITCH_START_ROUNDS, True)
    # print("lower boomerang distinguisher: {}\n".format(res))

    # # check boomerang switch
    # res = varify(SWITCH_INPUT_DIFF, SWITCH_OUTPUT_DIFF, 1, True)
    # print("boomerang switch: {}\n".format(res))


