import random
import math
import multiprocessing
from task_katan import checks
#from task_simon import checks


POOL = multiprocessing.Pool(processes=12)
ROUNDS = 60
WEIGHT = 22
# SWITCH_ROUNDS = 1    
# SWITCH_START_ROUNDS = int(ROUNDS/2)
# OFFSET = 0


def varify(in_diff, out_diff, rounds, boomerang, offset=0):
    test_n = 2**WEIGHT
    key = random.randint(0, 2**32)
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
    batch_size = 100000
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
                    boomerang
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
    data_file = open("check_list.txt", "r")

    data_list = []
    
    data = data_file.readline()
    while data != "":
        temps = data.split(",")
        datas = []
        for i in temps:
            if i.startswith("0x"):
                datas.append(int(i, 16))
            else:
                datas.append(int(i))
        datas.append(1)
        data_list.append(datas)
        data = data_file.readline()

    count = 0
    for dd in data_list:
        res = varify(dd[0], dd[3], ROUNDS, True)
        if res == "Invaild":
            count += 1
            print("in:{0}, out:{1}, sols:{2}\n".format(hex(dd[0]), hex(dd[3]), dd[4]))
        else:
            print("prob:{0}, sols:{1}".format(res, dd[4]))
    print("test cases: {0}, worng cases: {1}".format(len(data_list), count))
    # check boomerang distinguisher prob
    
    

    # # check upper tail
    # res = varify(INPUD_DIFF, SWITCH_INPUT_DIFF, SWITCH_START_ROUNDS, False)
    # print("boomerang upper trail: {}\n".format(res))

    # # check lower trail
    # res = varify(SWITCH_OUTPUT_DIFF, OUTPUT_DIFF, ROUNDS, False, SWITCH_START_ROUNDS+1)
    # print("boomerang lower trail: {}\n".format(res))

    # # check upper boomerang
    # res = varify(INPUD_DIFF, SWITCH_OUTPUT_DIFF, SWITCH_START_ROUNDS+1, True)
    # print("upper boomerang distinguisher: {}\n".format(res))

    # # check lower boomerang
    # res = varify(SWITCH_INPUT_DIFF, OUTPUT_DIFF, ROUNDS-SWITCH_START_ROUNDS, True)
    # print("lower boomerang distinguisher: {}\n".format(res))

    # check boomerang switch
    # res = varify(SWITCH_INPUT_DIFF, SWITCH_OUTPUT_DIFF, 1, True)
    # print("boomerang switch: {}\n".format(res))


