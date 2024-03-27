from wand.image import Image
import os

def pdf_to_jpg(pdf_path, output_folder):
    try:
        with Image(filename=pdf_path, resolution=300) as img:
            img.compression_quality = 100
            img.save(filename=os.path.join(output_folder, "output.jpg"))
        print("PDF转换为JPG成功！")
    except Exception as e:
        print(f"转换失败：{e}")

# 指定PDF文件路径和输出文件夹路径
pdf_path = "1.pdf"
output_folder = "output_folder"

# 调用函数将PDF转换为JPG
pdf_to_jpg(pdf_path, output_folder)