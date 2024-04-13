from PIL import Image
import os

def convert_to_grayscale(image_path):
    # 打开图像
    img = Image.open(image_path)
    
    # 转换为灰度图像
    gray_img = img.convert('L')
    
    # 保存覆盖原始文件
    gray_img.save(image_path)

def batch_convert_to_grayscale(folder_path):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            # 构造完整的文件路径
            image_path = os.path.join(folder_path, file_name)
            # 转换图像为灰度
            convert_to_grayscale(image_path)
            print(f"Converted {file_name} to grayscale.")

# 指定包含PNG图像的文件夹路径
folder_path = 'data'

# 批量将PNG图像转换为灰度图像
batch_convert_to_grayscale(folder_path)
