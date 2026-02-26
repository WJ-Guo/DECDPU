def format_sci_3(val_str):
    """
    将数值字符串转换为指定格式的科学计数法：
    - 保留7位小数
    - 指数部分为3位（例如 e+001）
    """
    try:
        val = float(val_str)
        # 使用标准科学计数法格式化，保留7位小数
        s = "{:.7e}".format(val)
        base, exponent = s.split('e')
        # 强制指数部分显示为3位数字（例如 +01 -> +001）
        return f"{base}e{int(exponent):+04d}"
    except ValueError:
        return val_str  # 如果无法转换为数字，保留原样


def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            # 去除首尾空白符并按逗号分隔
            parts = line.strip().split(',')
            if not parts:
                continue

            # 对每个数值应用格式化函数
            formatted_parts = [format_sci_3(p) for p in parts]

            # 使用两个空格连接并写入文件
            f_out.write("  ".join(formatted_parts) + "\n")


# 执行处理
input_filename = 'librasmdata.txt'
output_filename = 'librasmdata_formatted.txt'

try:
    process_file(input_filename, output_filename)
    print(f"处理完成！已生成文件: {output_filename}")

    # 打印前3行预览
    print("\n前3行数据预览:")
    with open(output_filename, 'r') as f:
        for _ in range(3):
            print(f.readline().strip())
except FileNotFoundError:
    print(f"错误: 找不到文件 {input_filename}")