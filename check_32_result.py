import random
import math
import multiprocessing
from new_task import check

POOL = multiprocessing.Pool(processes=12)

def varify():
    test_n = 2**22
    #test_n = 1
    key = random.randint(0, 2**32)
    records = {}
    count = 0
    result = 0
    task_list = []
    while count < test_n:
        x1 = random.randint(0, 2**32)
        if x1 in records:
            continue
        count += 1
        records[x1] = 1
        task_list.append(
            POOL.apply_async(check, args=(x1, key))
        )
    print(len(task_list))
    for task in task_list:
        result += task.get()
    if result == 0:
        print(0)
        return 
    prob = result/test_n
    print(math.log2(prob))

if __name__ == "__main__":
    varify()
