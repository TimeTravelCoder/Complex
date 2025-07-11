import os
import shutil

def flatten_directory(parent_folder: str):
    """
    将一个母文件夹内的所有子文件夹的内容提取到该母文件夹的根目录。

    :param parent_folder: 要处理的母文件夹的路径。
    """
    # 1. 验证路径是否存在且为文件夹
    if not os.path.isdir(parent_folder):
        print(f"错误：路径 '{parent_folder}' 不是一个有效的文件夹。")
        return

    print(f"开始处理文件夹：'{parent_folder}'")
    moved_files_count = 0
    
    # 2. 遍历所有子文件夹和文件
    # 使用 os.walk 从顶层开始遍历
    for root, dirs, files in os.walk(parent_folder):
        # 跳过母文件夹本身，我们只关心子文件夹里的文件
        if root == parent_folder:
            continue

        for filename in files:
            source_path = os.path.join(root, filename)
            dest_path = os.path.join(parent_folder, filename)

            # 3. 处理文件名冲突
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path):
                # 如果目标路径已存在文件，则重命名
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{counter}{ext}"
                dest_path = os.path.join(parent_folder, new_filename)
                counter += 1
            
            # 4. 移动文件
            try:
                shutil.move(source_path, dest_path)
                if original_dest_path != dest_path:
                    print(f"移动并重命名: '{source_path}' -> '{dest_path}'")
                else:
                    print(f"移动文件: '{source_path}' -> '{dest_path}'")
                moved_files_count += 1
            except Exception as e:
                print(f"移动文件 '{source_path}' 时出错: {e}")

    # 5. 删除空的子文件夹
    # 使用 topdown=False 从内到外遍历，这样可以先处理子文件夹再处理父文件夹
    deleted_folders_count = 0
    for root, dirs, files in os.walk(parent_folder, topdown=False):
        # 同样跳过母文件夹本身
        if root == parent_folder:
            continue
        
        # 检查文件夹是否为空
        if not os.listdir(root):
            try:
                os.rmdir(root)
                print(f"删除空文件夹: '{root}'")
                deleted_folders_count += 1
            except OSError as e:
                print(f"删除文件夹 '{root}' 时出错: {e}")

    print("\n--- 操作完成 ---")
    print(f"总共移动了 {moved_files_count} 个文件。")
    print(f"总共删除了 {deleted_folders_count} 个空文件夹。")


if __name__ == "__main__":
    print("--- 子文件夹内容提取工具 ---")
    print("此脚本会将您指定的母文件夹中，所有子文件夹里的文件移动到母文件夹根目录，并删除空的子文件夹。")
    print("注意：这是一个移动操作，不是复制。建议在操作前备份您的文件夹。")
    
    # 获取用户输入的母文件夹路径
    target_folder = input("请输入母文件夹的完整路径: ").strip()

    # 处理路径中可能存在的引号（例如从文件管理器复制路径时）
    if target_folder.startswith('"') and target_folder.endswith('"'):
        target_folder = target_folder[1:-1]
        
    flatten_directory(target_folder)