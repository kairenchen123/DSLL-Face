import json

# 打开JSON文件并加载数据
with open('/home/gzhu2023/gzhu2023/ckr/mmdetection/wider_out.json', 'r') as json_file:
    data = json.load(json_file)

# 现在，变量"data"包含了从JSON文件中加载的数据
print(len(data))
