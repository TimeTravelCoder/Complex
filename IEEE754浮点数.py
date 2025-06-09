import struct
import sys

def float_to_binary_hex(value, precision='single'):
    """
    将浮点数转为 IEEE 754 二进制和十六进制表示。
    
    参数:
        value: 要转换的浮点数。
        precision: 'single' 表示32位单精度，'double' 表示64位双精度。
    
    返回:
        (binary_str, hex_str): 二进制字符串和十六进制字符串
    """
    if precision == 'single':
        # 使用 struct 打包为 32 位 float（IEEE 754）
        packed = struct.pack('>f', value)  # 大端字节序
        total_bits = 32
    elif precision == 'double':
        # 使用 struct 打包为 64 位 float（IEEE 754）
        packed = struct.pack('>d', value)
        total_bits = 64
    else:
        raise ValueError("precision 参数必须是 'single' 或 'double'")

    # 转换为整数，再转二进制和十六进制
    int_repr = int.from_bytes(packed, byteorder='big')
    binary_str = format(int_repr, f'0{total_bits}b')
    hex_str = format(int_repr, f'0{total_bits // 4}X')

    return binary_str, hex_str

def print_result(value):
    """输出格式化的IEEE 754编码结果"""
    print(f"\n输入的十进制数值: {value}")

    binary_32, hex_32 = float_to_binary_hex(value, 'single')
    binary_64, hex_64 = float_to_binary_hex(value, 'double')

    print("\n单精度（32位）:")
    print(f"二进制表示: {binary_32}")
    print(f"十六进制表示: {hex_32}")

    print("\n双精度（64位）:")
    print(f"二进制表示: {binary_64}")
    print(f"十六进制表示: {hex_64}")

def main():
    """主函数：接收用户输入并调用处理函数"""
    try:
        user_input = input("请输入一个十进制数（整数或小数）: ").strip()
        value = float(user_input)
        print_result(value)
    except ValueError:
        print("输入无效，请输入一个有效的十进制数字。")

if __name__ == '__main__':
    main()
