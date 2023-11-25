import os

def process_txt_file(input_file, data):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    for line in lines[1:]:  # Start from the second line
        values = list(map(float, line.split()))
        width = values[2] - values[0]
        height = values[3] - values[1]
        data.append((width, height))

def process_folder(input_folder, output_file):
    data = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_folder, filename)
            process_txt_file(input_file, data)

    # Sort data based on width
    sorted_data = sorted(data, key=lambda x: x[0])

    with open(output_file, 'w') as outfile:
        for width, height in sorted_data:
            outfile.write(f'{width} {height}\n')

if __name__ == '__main__':
    input_folder = '/mnt/data2/ckr/darkface/Dark_train_gt'
    output_file = '/mnt/data2/ckr/mmdetection/work_dirs/pred/output.txt'

    process_folder(input_folder, output_file)
