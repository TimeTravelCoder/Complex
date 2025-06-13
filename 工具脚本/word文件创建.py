import os
import docx
from datetime import datetime

# 定义目标目录
target_dir = r"C:\Users\jiami\Documents\Word"

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取当前时间
current_time = datetime.now().strftime("%Y-%m-%d")

# 获取用户输入的文件名
file_name = input("请输入要创建的 Word 文件的名称（无需包含 .docx 扩展名）：")

# 拼接完整的文件路径，使用下划线分隔文件名和日期
file_path = os.path.join(target_dir, f"{file_name}__{current_time}.docx")

# 创建 Word 文档对象
doc = docx.Document()



# 保存文档到指定路径
doc.save(file_path)

# 输出完整的文件名
print(f"已在 {target_dir} 目录下成功创建文件 {file_name}{current_time}.docx")
    