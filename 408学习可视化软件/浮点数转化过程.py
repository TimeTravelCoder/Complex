import struct

def float_to_bits(x, precision):
    if precision == 32:
        packed = struct.pack('!f', x)
        unpacked = struct.unpack('!I', packed)[0]
        bits = format(unpacked, '032b')
    elif precision == 64:
        packed = struct.pack('!d', x)
        unpacked = struct.unpack('!Q', packed)[0]
        bits = format(unpacked, '064b')
    else:
        raise ValueError("仅支持 32 或 64 位精度")
    
    return bits, hex(unpacked)[2:].upper().zfill(precision // 4)

def format_single_precision(x):
    bits, hex_val = float_to_bits(x, 32)
    return bits[0], bits[1:9], bits[9:], bits, hex_val

def format_double_precision(x):
    bits, hex_val = float_to_bits(x, 64)
    return bits[0], bits[1:12], bits[12:], bits, hex_val

def main():
    x = float(input("请输入一个浮点数: "))
    
    print("单精度转换:")
    sign, exponent, mantissa, bits, hex_val = format_single_precision(x)
    print(f"符号位: {sign}")
    print(f"指数位: {exponent}")
    print(f"尾数位: {mantissa}")
    print(f"完整的32位二进制表示: {bits}")
    print(f"十六进制表示: 0x{hex_val}")
    
    print("\n双精度转换:")
    sign, exponent, mantissa, bits, hex_val = format_double_precision(x)
    print(f"符号位: {sign}")
    print(f"指数位: {exponent}")
    print(f"尾数位: {mantissa}")
    print(f"完整的64位二进制表示: {bits}")
    print(f"十六进制表示: 0x{hex_val}")

if __name__ == "__main__":
    main()
