import numpy as np
import matplotlib.pyplot as plt

WORD_SIZE = 2 ** 1

title = ""

DEL_IN = 0


def compute_btc(delta_in: int):
    global title
    title = "BTC"
    for delta_comma in range(WORD_SIZE):
        for x_1 in range(WORD_SIZE):
            for x_1_comma in range(WORD_SIZE):
                for x_3 in range(WORD_SIZE):
                    for x_3_comma in range(WORD_SIZE):
                        y_1 = x_1 & x_1_comma
                        y_3 = x_3 & x_3_comma
                        y_2 = (x_1 ^ delta_in) & (x_1_comma ^ delta_comma)
                        y_4 = (x_3 ^ delta_in) & (x_3_comma ^ delta_comma)
                        delta_1_3 = y_1 ^ y_3
                        delta_2_4 = y_2 ^ y_4
                        if delta_1_3 == delta_2_4:
                            BCT[delta_comma][delta_1_3] += 1


def compute_ddt(delta_in: int):
    global title
    title = "DDT"
    for delta_dot in range(WORD_SIZE):
        for x_1 in range(WORD_SIZE):
            x_2 = x_1 ^ delta_in
            x_1_dot = x_1 ^ delta_dot
            x_2_dot = x_2 ^ delta_dot
            delta_y = (x_1 & x_1_dot) ^ (x_2 & x_2_dot)
            BCT[delta_dot][delta_y] += 1


def draw_ddt_table():
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    color_arr = list()
    # for i in range(16):
    #     tl = list()
    #     for j in range(16):
    #         if DIFFER_APPROXIMATION_TABLE[i][j] == 2:
    #             tl.append("red")
    #         elif DIFFER_APPROXIMATION_TABLE[i][j] == 4:
    #             tl.append('pink')
    #         else:
    #             tl.append('yellow')
    #     color_arr.append(tl)
    ax.table(cellText=BCT
             , colLabels=[i for i in range(WORD_SIZE)]
             , rowLabels=[i for i in range(WORD_SIZE)]
             # , colColours=['green' for i in range(16)]
             # , rowColours=['green' for i in range(16)]
             # , cellColours=color_arr
             , loc='center')
    fig.tight_layout()
    plt.suptitle(title + "\nfixed param: Delta_in=" + str(DEL_IN))
    plt.show()


BCT = np.zeros((WORD_SIZE, WORD_SIZE))
compute_btc(DEL_IN)
draw_ddt_table()
BCT = np.zeros((WORD_SIZE, WORD_SIZE))
compute_ddt(DEL_IN)
draw_ddt_table()
