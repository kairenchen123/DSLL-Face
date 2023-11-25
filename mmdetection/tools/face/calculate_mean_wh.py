def sort_file_by_first_column(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 将每一行的数字以空格分隔，并转换为浮点数
    lines = [list(map(float, line.split())) for line in lines]

    # 按每行的第一个值进行排序
    sorted_lines = sorted(lines, key=lambda x: x[0])

    # 将排序后的结果写入新的txt文件
    with open(output_file, 'w') as f:
        for line in sorted_lines:
            f.write(" ".join(map(str, line)) + "\n")

if __name__ == "__main__":
    # 输入文件路径
    input_file = "/mnt/data2/ckr/mmdetection/work_dirs/pred/file.txt"

    # 输出文件路径
    output_file = "/mnt/data2/ckr/mmdetection/work_dirs/pred/sorted_file.txt"

    # 对文件按第一列的值进行排序
    sort_file_by_first_column(input_file, output_file)
