import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import plotly.io as pio
from typing import List, Dict, Any, Optional
import random
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 设置中文字体支持
pio.templates.default = "plotly_white"

class Process:
    """表示一个进程"""
    def __init__(self, pid: str, arrival_time: int, run_time: int, priority: int = None):
        self.pid = pid                   # 进程ID
        self.arrival_time = arrival_time # 到达时间
        self.run_time = run_time         # 运行时间
        self.priority = priority         # 优先级（数值越小优先级越高）
        self.remaining_time = run_time   # 剩余运行时间
        self.start_time = None           # 首次执行时间
        self.completion_time = None      # 完成时间
        self.waiting_time = None         # 等待时间
        self.turnaround_time = None      # 周转时间
        self.response_time = None        # 响应时间

    def __repr__(self):
        return f"Process(pid={self.pid}, arrival_time={self.arrival_time}, run_time={self.run_time})"

class Scheduler:
    """调度器基类"""
    def __init__(self, processes: List[Process]):
        self.processes = processes.copy()
        self.schedule = []  # 存储调度结果 [(进程ID, 开始时间, 结束时间)]
        self.time = 0       # 当前时间
        self.completed = 0  # 已完成的进程数
        self.total_processes = len(processes)
        self.ready_queue = []  # 就绪队列

    def get_next_process(self) -> Optional[Process]:
        """获取下一个要执行的进程，由子类实现"""
        raise NotImplementedError

    def execute(self) -> None:
        """执行调度算法"""
        while self.completed < self.total_processes:
            # 更新就绪队列
            self.update_ready_queue()

            # 获取下一个要执行的进程
            current_process = self.get_next_process()

            if current_process is None:
                # 没有可执行的进程，时间推进
                self.time += 1
                continue

            # 执行一个时间单位
            current_process.remaining_time -= 1

            # 记录首次执行时间（响应时间）
            if current_process.start_time is None:
                current_process.start_time = self.time

            # 更新调度记录
            if not self.schedule or self.schedule[-1][0] != current_process.pid:
                self.schedule.append((current_process.pid, self.time, self.time + 1))
            else:
                # 扩展当前进程的执行时间
                pid, start, end = self.schedule.pop()
                self.schedule.append((pid, start, self.time + 1))

            # 时间推进
            self.time += 1

            # 检查进程是否完成
            if current_process.remaining_time == 0:
                current_process.completion_time = self.time
                current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.run_time
                current_process.response_time = current_process.start_time - current_process.arrival_time
                self.completed += 1

    def update_ready_queue(self) -> None:
        """更新就绪队列，将到达的进程加入队列"""
        for process in self.processes:
            if process.arrival_time <= self.time and process.remaining_time > 0 and process not in self.ready_queue:
                self.ready_queue.append(process)

    def get_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        if self.completed == 0:
            return {
                "CPU利用率": 0,
                "吞吐量": 0,
                "平均周转时间": 0,
                "平均等待时间": 0,
                "平均响应时间": 0
            }

        # CPU利用率 = 执行时间 / 总时间
        total_run_time = sum(p.run_time for p in self.processes)
        cpu_utilization = total_run_time / self.time

        # 吞吐量 = 完成的进程数 / 总时间
        throughput = self.completed / self.time

        # 平均周转时间
        completed_processes = [p for p in self.processes if p.completion_time is not None]
        avg_turnaround_time = sum(p.turnaround_time for p in completed_processes) / self.completed

        # 平均等待时间
        avg_waiting_time = sum(p.waiting_time for p in completed_processes) / self.completed

        # 平均响应时间
        avg_response_time = sum(p.response_time for p in completed_processes) / self.completed

        return {
            "CPU利用率": cpu_utilization,
            "吞吐量": throughput,
            "平均周转时间": avg_turnaround_time,
            "平均等待时间": avg_waiting_time,
            "平均响应时间": avg_response_time
        }

class FCFS(Scheduler):
    """先来先服务调度器"""
    def get_next_process(self) -> Optional[Process]:
        if not self.ready_queue:
            return None
        # 按到达时间排序，如果到达时间相同则按进程ID排序
        self.ready_queue.sort(key=lambda p: (p.arrival_time, p.pid))
        return self.ready_queue.pop(0)

class SJF(Scheduler):
    """短作业优先调度器（非抢占式）"""
    def get_next_process(self) -> Optional[Process]:
        if not self.ready_queue:
            return None
        # 按剩余运行时间排序，如果相同则按到达时间排序
        self.ready_queue.sort(key=lambda p: (p.remaining_time, p.arrival_time))
        return self.ready_queue.pop(0)

class Priority(Scheduler):
    """优先级调度器（非抢占式）"""
    def get_next_process(self) -> Optional[Process]:
        if not self.ready_queue:
            return None
        # 按优先级排序（数值越小优先级越高），如果优先级相同则按到达时间排序
        self.ready_queue.sort(key=lambda p: (p.priority, p.arrival_time))
        return self.ready_queue.pop(0)

class RoundRobin(Scheduler):
    """时间片轮转调度器"""
    def __init__(self, processes: List[Process], time_quantum: int):
        super().__init__(processes)
        self.time_quantum = time_quantum  # 时间片大小
        self.current_time_quantum = 0     # 当前时间片计数器

    def get_next_process(self) -> Optional[Process]:
        if not self.ready_queue:
            return None

        # 如果当前时间片用完或队列为空，将当前进程移到队尾
        if self.current_time_quantum >= self.time_quantum:
            self.ready_queue.append(self.ready_queue.pop(0))
            self.current_time_quantum = 0

        current_process = self.ready_queue[0]
        self.current_time_quantum += 1

        # 如果进程执行完毕，从队列中移除
        if current_process.remaining_time == 0:
            self.ready_queue.pop(0)
            self.current_time_quantum = 0

        return current_process

def generate_test_processes(num_processes: int = 5, max_arrival_time: int = 10, max_run_time: int = 10) -> List[Process]:
    """生成测试进程数据"""
    processes = []
    for i in range(num_processes):
        pid = f"P{i+1}"
        arrival_time = random.randint(0, max_arrival_time)
        run_time = random.randint(1, max_run_time)
        priority = random.randint(1, 5)  # 优先级1-5
        processes.append(Process(pid, arrival_time, run_time, priority))
    return processes

def input_processes() -> List[Process]:
    """手动输入进程信息"""
    processes = []
    while True:
        pid = input("请输入进程名（输入q结束）：")
        if pid.lower() == 'q':
            break

        try:
            arrival_time = int(input("请输入到达时间："))
            run_time = int(input("请输入运行时间："))
            priority_input = input("请输入优先级（可选，直接回车跳过）：")
            priority = int(priority_input) if priority_input else None

            processes.append(Process(pid, arrival_time, run_time, priority))
            print(f"进程 {pid} 已添加")
        except ValueError:
            print("输入无效，请输入有效的数字！")

    return processes

def generate_gantt_chart(schedule: List[Dict], title: str, metrics: Dict[str, float]) -> None:
    """生成甘特图"""
    # 为不同的进程分配不同的颜色
    pids = list({entry['Task'] for entry in schedule})
    colors = {}

    # 使用Plotly的默认颜色方案
    plotly_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    for i, pid in enumerate(pids):
        colors[pid] = plotly_colors[i % len(plotly_colors)]

    # 创建甘特图数据，使用字符串表示时间以避免Unix时间戳问题
    gantt_data = []
    for entry in schedule:
        gantt_data.append({
            'Task': entry['Task'],
            'Start': entry['Start'],  # 使用整数表示时间
            'Finish': entry['Finish'],  # 使用整数表示时间
            'Resource': entry['Resource']
        })

    # 创建包含甘特图和表格的子图
    fig = make_subplots(rows=2, cols=1, 
                        specs=[[{'type': 'xy'}], [{'type': 'table'}]],
                        row_heights=[0.8, 0.2])

    gantt_fig = ff.create_gantt(
        gantt_data,
        colors=colors,
        index_col='Task',
        show_colorbar=True,
        group_tasks=True,
        title=title,
        showgrid_x=True,
        showgrid_y=True,
        bar_width=0.4,  # 调整条形宽度
        height=400
    )

    for trace in gantt_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # 确保时间轴从0开始
    all_times = [entry['Start'] for entry in gantt_data] + [entry['Finish'] for entry in gantt_data]
    min_time = 0  # 强制时间轴从0开始
    max_time = max(all_times)

    # 设置x轴范围和刻度
    fig.update_layout(
        font=dict(family="SimHei, WenQuanYi Micro Hei, Heiti TC", size=12),
        title=dict(font=dict(family="SimHei, WenQuanYi Micro Hei, Heiti TC", size=16)),
        xaxis_title="时间",
        yaxis_title="进程",
        xaxis=dict(
            type='linear',  # 确保使用线性轴而非日期轴
            tickmode='array',  # 使用数组指定刻度位置
            tickvals=list(range(min_time, max_time + 1)),  # 从0到最大时间的整数刻度
            range=[min_time - 0.5, max_time + 0.5]  # 设置x轴范围
        )
    )

    # 创建表格数据（使用中文指标名称）
    header = ['指标', '数值']
    cells = [
        list(metrics.keys()),
        [f"{val:.2%}" if key == "CPU利用率" else f"{val:.2f}" for key, val in metrics.items()]
    ]

    table_trace = go.Table(
        header=dict(values=header),
        cells=dict(values=cells)
    )

    fig.add_trace(table_trace, row=2, col=1)

    # 显示图表
    fig.show()

def main():
    """主函数"""
    print("===== 操作系统调度算法可视化工具 =====")

    # 选择输入方式
    print("\n请选择输入方式：")
    print("1. 手动输入进程数据")
    print("2. 生成测试数据")

    while True:
        try:
            input_choice = int(input("请输入选择 (1-2)："))
            if input_choice == 1:
                processes = input_processes()
            elif input_choice == 2:
                try:
                    num_processes = int(input("请输入要生成的进程数量（默认5）：") or "5")
                    max_arrival_time = int(input("请输入最大到达时间（默认10）：") or "10")
                    max_run_time = int(input("请输入最大运行时间（默认10）：") or "10")
                    processes = generate_test_processes(num_processes, max_arrival_time, max_run_time)
                    print("\n生成的进程数据：")
                    for p in processes:
                        print(f"进程: {p.pid}, 到达时间: {p.arrival_time}, 运行时间: {p.run_time}, 优先级: {p.priority}")
                except ValueError:
                    print("输入无效，使用默认参数生成测试数据...")
                    processes = generate_test_processes()
            else:
                print("无效选择，请输入1-2之间的数字！")
                continue
            break
        except ValueError:
            print("无效输入，请输入数字！")

    if not processes:
        print("至少需要一个进程！")
        return

    # 选择调度算法
    print("\n请选择调度算法：")
    print("1. FCFS (先来先服务)")
    print("2. SJF (短作业优先)")
    print("3. 优先级调度")
    print("4. 时间片轮转 (Round Robin)")

    while True:
        try:
            choice = int(input("请输入选择 (1-4)："))
            if choice not in [1, 2, 3, 4]:
                print("无效选择，请输入1-4之间的数字！")
                continue

            if choice == 4:
                time_quantum = int(input("请输入时间片大小："))
                scheduler = RoundRobin(processes.copy(), time_quantum)
                algorithm_name = f"时间片轮转 (时间片={time_quantum})"
            elif choice == 1:
                scheduler = FCFS(processes.copy())
                algorithm_name = "FCFS (先来先服务)"
            elif choice == 2:
                scheduler = SJF(processes.copy())
                algorithm_name = "SJF (短作业优先)"
            else:  # choice == 3
                # 检查所有进程是否都有优先级
                if any(p.priority is None for p in processes):
                    print("错误：优先级调度需要所有进程都有优先级！")
                    continue
                scheduler = Priority(processes.copy())
                algorithm_name = "优先级调度"

            break
        except ValueError:
            print("无效输入，请输入数字！")

    # 执行调度算法
    scheduler.execute()

    # 准备甘特图数据
    gantt_data = []
    for pid, start, end in scheduler.schedule:
        gantt_data.append({
            'Task': pid,
            'Start': start,
            'Finish': end,
            'Resource': pid
        })

    # 计算性能指标
    metrics = scheduler.get_metrics()

    # 显示结果
    print(f"\n===== {algorithm_name} 调度结果 =====")
    print("\n进程执行顺序：")
    for pid, start, end in scheduler.schedule:
        print(f"时间 [{start}-{end}]: 进程 {pid}")

    print("\n各进程性能指标：")
    print("进程\t到达时间\t运行时间\t完成时间\t周转时间\t等待时间\t响应时间")
    for p in sorted(processes, key=lambda x: x.arrival_time):
        print(f"{p.pid}\t{p.arrival_time}\t\t{p.run_time}\t\t{p.completion_time}\t\t{p.turnaround_time}\t\t{p.waiting_time}\t\t{p.response_time}")

    print("\n系统整体性能指标：")
    print(f"CPU利用率: {metrics['CPU利用率']:.2%}")
    print(f"吞吐量: {metrics['吞吐量']:.2f} 个进程/单位时间")
    print(f"平均周转时间: {metrics['平均周转时间']:.2f}")
    print(f"平均等待时间: {metrics['平均等待时间']:.2f}")
    print(f"平均响应时间: {metrics['平均响应时间']:.2f}")

    # 生成甘特图
    generate_gantt_chart(gantt_data, f"{algorithm_name} 调度甘特图", metrics)

if __name__ == "__main__":
    main()