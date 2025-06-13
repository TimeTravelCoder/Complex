import os
def print_file_structure(folder_path, level=0):
    try:
        items = os.listdir(folder_path)
        for item in items:
            item_path = os.path.join(folder_path, item)
            prefix = "|   " * level + "|-- " if level > 0 else ""
            if os.path.isdir(item_path):
                print(prefix + item + "/")
                print_file_structure(item_path, level + 1)
            else:
                print(prefix + item)
    except PermissionError:
        print(f"权限不足，无法访问 {folder_path}")


if __name__ == "__main__":
    target_folder = input("请输入要查看文件结构的文件夹路径：")
    print_file_structure(target_folder)