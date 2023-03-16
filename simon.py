import numpy


def create_and_bct(non_linear, involve_bits_length: int):
    """
    use specified non-part function to create AND-BCT
    :param non_linear: non-liner function
    :param involve_bits_length: length of a number involving AND-operation
    :return: AND-BCT
    """
    table_size = 2 ** 8
    and_bct = numpy.zeros((table_size, table_size), dtype=int)
    for delta_in in range(table_size):
        for nabla_out in range(table_size):
            for x1 in range(table_size):
                x2 = x1 ^ delta_in
                x3 = x1 ^ nabla_out
                x4 = x1 ^ delta_in ^ nabla_out
                y1 = non_linear(x1, involve_bits_length)
                y2 = non_linear(x2, involve_bits_length)
                y3 = non_linear(x3, involve_bits_length)
                y4 = non_linear(x4, involve_bits_length)
                if y1 ^ y2 ^ y3 ^ y4 == 0:
                    and_bct[delta_in][nabla_out] += 1
    return and_bct


def general_and_operation(x: int, involve_bits_length: int):
    x1 = circular_shift_left(x, 1, 16)
    x2 = circular_shift_left(x, 8, 16)
    return x1 & x2


def circular_shift_left(int_value, k, bit=8):
    bit_string = '{:0%db}' % bit
    bin_value = bit_string.format(int_value)  # 8 bit binary
    bin_value = bin_value[k:] + bin_value[:k]
    int_value = int(bin_value, 2)
    return int_value


def circular_shift_right(int_value, k, bit=8):
    bit_string = '{:0%db}' % bit
    bin_value = bit_string.format(int_value)  # 8 bit binary
    bin_value = bin_value[-k:] + bin_value[:-k]
    int_value = int(bin_value, 2)
    return int_value


bct = create_and_bct(general_and_operation, 0)
print()
