from PIL import Image
import os

def convert_to_binary(image_path):
    # 打开图像
    img = Image.open(image_path)
    
    # 将255像素值替换为1
    img = img.point(lambda p: p == 255 and 1)
    
    # 保存覆盖原始文件
    img.save(image_path)

def batch_convert_to_binary(folder_path):
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            # 构造完整的文件路径
            image_path = os.path.join(folder_path, file_name)
            # 转换图像为二值图像
            convert_to_binary(image_path)
            print(f"Converted {file_name} to binary.")

# 指定包含PNG图像的文件夹路径
folder_path = 'data'

# 批量将PNG图像转换为二值图像
batch_convert_to_binary(folder_path)
