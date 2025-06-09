import os
from datetime import datetime
import docx
import pandas as pd
from pptx import Presentation

# 定义目标目录
target_dir = r"C:\Users\jiami\Documents\Word"

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取用户输入的文件名
file_name = input("请输入要创建的文件的名称（无需包含扩展名）：")
current_time = datetime.now().strftime("%Y-%m-%d")
# 让用户选择文件格式
print("请选择文件格式：")
print("1. Word (.docx)")
print("2. TXT (.txt)")
print("3. Excel (.xlsx)")
print("4. PPT (.pptx)")
print("5. Markdown (.md)")
choice = input("请输入对应的数字 (1 - 5): ")

# 拼接完整的文件路径
if choice == '1':
    file_ext = ".docx"
elif choice == '2':
    file_ext = ".txt"
elif choice == '3':
    file_ext = ".xlsx"
elif choice == '4':
    file_ext = ".pptx"
elif choice == '5':
    file_ext = ".md"
else:
    print("无效的选择，脚本退出。")
    exit(1)

file_path = os.path.join(target_dir, f"{file_name}__{current_time}{file_ext}")

# 根据用户选择创建不同格式的文件
if choice == '1':
    # 创建 Word 文档对象
    doc = docx.Document()
    # 保存文档到指定路径
    doc.save(file_path)
elif choice == '2':
    # 创建并写入 TXT 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        pass
elif choice == '3':
    # 创建空的 DataFrame
    df = pd.DataFrame()
    # 保存为 Excel 文件
    df.to_excel(file_path, index=False)
elif choice == '4':
    # 创建 PPT 对象
    prs = Presentation()
    # 保存 PPT 文件
    prs.save(file_path)
elif choice == '5':
    # 创建并写入 Markdown 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        pass

print(f"已在 {target_dir} 目录下成功创建文件 {file_name}{file_ext}")
    