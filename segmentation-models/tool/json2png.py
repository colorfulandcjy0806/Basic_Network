import json
import numpy as np
from PIL import Image, ImageDraw
import os

def process_json_file(json_file_path, output_folder):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    image = Image.new('RGB', (data['imageWidth'], data['imageHeight']), 'black')
    draw = ImageDraw.Draw(image)

    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            polygon = [(point[0], point[1]) for point in shape['points']]
            draw.polygon(polygon, fill='white')

    base_name = os.path.basename(json_file_path)
    output_file_name = os.path.splitext(base_name)[0] + '.png'
    output_file_path = os.path.join(output_folder, output_file_name)

    image.save(output_file_path)


# 定义JSON文件夹和输出PNG文件夹路径
json_folder = 'json'
output_folder = 'mask'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file in os.listdir(json_folder):
    if file.endswith(".json"):
        json_file_path = os.path.join(json_folder, file)
        process_json_file(json_file_path, output_folder)

print("所有文件处理完成。")
