import subprocess
import os


def list_conda_environments():
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True, check=True)
        output = result.stdout
        lines = output.strip().split('\n')[2:]
        envs = []
        print("可用的 Conda 环境：")
        for i, line in enumerate(lines, start=1):
            parts = line.split()
            env_name = parts[0]
            envs.append(env_name)
            print(f"{i}. {env_name}")
        return envs
    except subprocess.CalledProcessError as e:
        print(f"查看 Conda 环境时出错：无法执行 'conda env list' 命令，错误信息：{e.stderr}")
        return []
    except Exception as e:
        print(f"查看 Conda 环境时出现未知错误：{e}")
        return []


def create_conda_environment():
    env_name = input("请输入要创建的虚拟环境名称：")
    try:
        subprocess.run(['conda', 'create', '-n', env_name], check=True)
        print(f"成功创建虚拟环境 {env_name}。")
    except subprocess.CalledProcessError as e:
        print(f"创建虚拟环境时出错：无法执行 'conda create -n {env_name}' 命令，错误信息：{e.stderr}")


def delete_conda_environment(envs):
    while True:
        try:
            num = int(input("请输入要删除的环境序号：")) - 1
            if 0 <= num < len(envs):
                env_to_delete = envs[num]
                confirm = input(f"确认要删除环境 {env_to_delete} 吗？(y/n): ")
                if confirm.lower() == 'y':
                    subprocess.run(['conda', 'env', 'remove', '-n', env_to_delete], check=True)
                    print(f"环境 {env_to_delete} 已删除。")
                else:
                    print("删除操作已取消。")
                break
            else:
                print("输入的序号无效，请重新输入。")
        except ValueError:
            print("输入的不是有效的数字，请重新输入。")
        except subprocess.CalledProcessError as e:
            print(f"删除环境时出错：无法执行 'conda env remove -n {env_to_delete}' 命令，错误信息：{e.stderr}")


def activate_conda_environment(envs):
    while True:
        try:
            num = int(input("请输入要激活的环境序号：")) - 1
            if 0 <= num < len(envs):
                env_to_activate = envs[num]
                if os.name == 'nt':  # Windows 系统
                    activate_command = f'conda activate {env_to_activate}'
                else:  # Linux 或 macOS 系统
                    activate_command = f'source activate {env_to_activate}'
                print(f"要激活环境 {env_to_activate}，请在终端中运行：{activate_command}")
                break
            else:
                print("输入的序号无效，请重新输入。")
        except ValueError:
            print("输入的不是有效的数字，请重新输入。")


def install_package_in_conda_environment(envs):
    while True:
        try:
            num = int(input("请输入要安装库的环境序号：")) - 1
            if 0 <= num < len(envs):
                env = envs[num]
                package = input("请输入要安装的库名：")
                try:
                    subprocess.run(['conda', 'install', '-n', env, package], check=True)
                    print(f"已在环境 {env} 中使用 conda 安装 {package}。")
                except subprocess.CalledProcessError:
                    print(f"使用 conda 在环境 {env} 中安装 {package} 失败，尝试使用 pip 安装...")
                    activate_cmd = f'conda activate {env}' if os.name == 'nt' else f'source activate {env}'
                    full_pip_cmd = f'{activate_cmd} && pip install {package}'
                    try:
                        subprocess.run(full_pip_cmd, shell=True, check=True)
                        print(f"已在环境 {env} 中使用 pip 安装 {package}。")
                    except subprocess.CalledProcessError as e:
                        print(f"使用 pip 在环境 {env} 中安装 {package} 也失败了，错误信息：{e.stderr}")
                break
            else:
                print("输入的序号无效，请重新输入。")
        except ValueError:
            print("输入的不是有效的数字，请重新输入。")


def get_valid_choice(prompt, min_choice, max_choice):
    while True:
        try:
            choice = int(input(prompt))
            if min_choice <= choice <= max_choice:
                return choice
            else:
                print(f"无效的选择，请输入 {min_choice} 到 {max_choice} 之间的数字。")
        except ValueError:
            print("输入的不是有效的数字，请重新输入。")


if __name__ == "__main__":
    envs = []
    while True:
        print("\nConda 环境管理菜单：")
        print("1. 查看虚拟环境")
        print("2. 创建虚拟环境")
        print("3. 根据序号删除虚拟环境")
        print("4. 激活虚拟环境")
        print("5. 在指定虚拟环境安装库")
        print("6. 退出")
        choice = get_valid_choice("请输入你的选择（1 - 6）：", 1, 6)
        if choice == 1:
            envs = list_conda_environments()
        elif choice == 2:
            create_conda_environment()
        elif choice == 3:
            if not envs:
                envs = list_conda_environments()
            if envs:
                delete_conda_environment(envs)
        elif choice == 4:
            if not envs:
                envs = list_conda_environments()
            if envs:
                activate_conda_environment(envs)
        elif choice == 5:
            if not envs:
                envs = list_conda_environments()
            if envs:
                install_package_in_conda_environment(envs)
        elif choice == 6:
            print("退出脚本。")
            break
    