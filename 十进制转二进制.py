def int_to_bin(x, width):
    if x >= 0:
        sign = '0'
        bin_val = bin(x)[2:].zfill(width - 1)
    else:
        sign = '1'
        bin_val = bin(abs(x))[2:].zfill(width - 1)
    return sign + bin_val


def get_ones_complement(bin_str):
    return bin_str[0] + ''.join('1' if b == '0' else '0' for b in bin_str[1:])


def get_twos_complement(bin_str):
    # 补码 = 反码 + 1
    sign = bin_str[0]
    inverted = get_ones_complement(bin_str)[1:]
    increment = bin(int(inverted, 2) + 1)[2:].zfill(len(inverted))
    if len(increment) > len(inverted):  # 溢出截断
        increment = increment[-len(inverted):]
    return sign + increment


def get_bias_code(x, width):
    bias = 2**(width - 1) - 1
    biased_val = x + bias
    if biased_val < 0 or biased_val >= 2**width:
        return f"<溢出>"
    return bin(biased_val)[2:].zfill(width)


def print_encodings(x, width=8):
    print(f"十进制输入 X = {x}")
    print(f"位宽：{width}位")

    # 真值二进制（带符号位）
    true_bin = int_to_bin(x, width)
    print(f"真值二进制：{true_bin}")

    # 原码
    sign = '0' if x >= 0 else '1'
    magnitude = bin(abs(x))[2:].zfill(width - 1)
    original_code = sign + magnitude
    print(f"原码：      {original_code}")

    # 反码
    if x >= 0:
        ones_comp = original_code
    else:
        ones_comp = get_ones_complement(original_code)
    print(f"反码：      {ones_comp}")

    # 补码
    if x >= 0:
        twos_comp = original_code
    else:
        twos_comp = get_twos_complement(original_code)
    print(f"补码：      {twos_comp}")

    # 移码
    bias_code = get_bias_code(x, width)
    print(f"移码：      {bias_code}")


if __name__ == "__main__":
    try:
        x = int(input("请输入一个整数 X："))
        width = int(input("请输入位宽（如8或16）："))
        if width < 2:
            print("位宽必须大于等于2。")
        else:
            print_encodings(x, width)
    except ValueError:
        print("请输入有效的整数。")
