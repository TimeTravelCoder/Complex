
# -*- coding: utf-8 -*-

import random
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import sys
import platform

class Process:
    """进程类，包含进程的各项属性"""
    def __init__(self, pid: int, arrival_time: int, run_time: int, priority: Optional[int] = None):
        self.pid = pid
        self.arrival_time = arrival_time
        self.run_time = run_time
        self.priority = priority
        self.remaining_time = run_time
        self.start_time = None  # 首次执行的时间
        self.completion_time = None  # 完成时间
        self.waiting_time = None  # 等待时间 = 周转时间 - 运行时间
        self.turnaround_time = None  # 周转时间 = 完成时间 - 到达时间
        self.response_time = None  # 响应时间 = 首次执行时间 - 到达时间
        self.current_queue = 0  # 用于多级队列和多级反馈队列

    def __str__(self):
        return f"进程 {self.pid} (到达:{self.arrival_time}, 运行:{self.run_time}, 优先级:{self.priority if self.priority is not None else 'N/A'})"

    def calculate_metrics(self):
        """计算进程的性能指标"""
        if self.completion_time is not None:
            self.turnaround_time = self.completion_time - self.arrival_time
            self.waiting_time = self.turnaround_time - self.run_time
            self.response_time = self.start_time - self.arrival_time


class Scheduler(ABC):
    """调度器基类"""
    def __init__(self, processes: List[Process], name: str):
        self.processes = processes.copy()  # 所有进程
        self.ready_queue = []  # 就绪队列
        self.current_time = 0  # 当前时间
        self.execution_sequence = []  # 执行序列记录
        self.completed_processes = []  # 已完成的进程
        self.current_process = None  # 当前正在运行的进程
        self.name = name  # 调度算法名称
        self.config = {}  # 调度算法配置参数

    def update_ready_queue(self) -> None:
        """更新就绪队列：将已到达但未在就绪队列且未完成的进程加入就绪队列"""
        for process in self.processes:
            if (process.arrival_time <= self.current_time and 
                process.remaining_time > 0 and 
                process not in self.ready_queue and 
                process not in self.completed_processes and
                process != self.current_process):
                self.ready_queue.append(process)

    def execute_process(self, process: Process, duration: int) -> bool:
        """执行进程一个时间单位，返回是否完成执行"""
        # 记录首次执行时间
        if process.start_time is None:
            process.start_time = self.current_time

        process.remaining_time -= duration
        end_time = self.current_time + duration
        completed = process.remaining_time <= 0

        status = "完成" if completed else "执行"
        self.execution_sequence.append({
            "pid": process.pid,
            "start_time": self.current_time,
            "end_time": end_time,
            "status": status
        })

        # 更新当前时间
        self.current_time = end_time

        # 如果进程已完成
        if completed:
            process.completion_time = end_time
            if process.remaining_time < 0:  # 处理可能的超出执行时间
                process.completion_time += process.remaining_time
                self.current_time += process.remaining_time
                self.execution_sequence[-1]["end_time"] = process.completion_time

            process.calculate_metrics()
            self.completed_processes.append(process)
            if process in self.ready_queue:
                self.ready_queue.remove(process)
            return True

        return False

    def is_preemptive(self) -> bool:
        """判断调度算法是否是抢占式的"""
        return False

    @abstractmethod
    def select_process(self) -> Optional[Process]:
        """选择下一个要执行的进程"""
        pass

    def run(self) -> Dict:
        """运行调度算法，模拟整个调度过程"""
        while len(self.completed_processes) < len(self.processes):
            # 更新就绪队列
            self.update_ready_queue()

            # 如果就绪队列为空，时间向前推进到下一个进程到达时间
            if not self.ready_queue and not self.current_process:
                next_arrival = min([p.arrival_time for p in self.processes 
                                    if p.arrival_time > self.current_time], default=None)
                if next_arrival is not None:
                    self.current_time = next_arrival
                    continue
                else:
                    # 如果没有更多进程到达，且就绪队列为空，则终止调度
                    break

            # 选择要执行的进程
            next_process = self.select_process()
            
            if next_process:
                # 如果是抢占式且当前有运行中的进程，检查是否需要抢占
                if self.is_preemptive() and self.current_process and self.current_process.remaining_time > 0:
                    # 记录抢占
                    self.execution_sequence.append({
                        "pid": self.current_process.pid,
                        "start_time": self.current_time,
                        "end_time": self.current_time,
                        "status": "抢占"
                    })
                    # 将当前进程放回就绪队列
                    if self.current_process not in self.ready_queue:
                        self.ready_queue.append(self.current_process)

                # 执行所选进程
                self.current_process = next_process
                if next_process in self.ready_queue:
                    self.ready_queue.remove(next_process)
                
                # 获取当前进程应执行的时间单位
                execution_time = self.get_execution_time(next_process)
                
                # 执行进程
                completed = self.execute_process(next_process, execution_time)
                
                # 如果进程完成，当前进程设为None；否则根据算法决定是否继续执行
                if completed:
                    self.current_process = None
                elif not self.is_preemptive():
                    # 非抢占式算法在进程未完成时继续执行
                    pass
                else:
                    # 抢占式算法在时间片结束后重新考虑就绪队列
                    # 将未完成的进程重新放入就绪队列（如果是时间片轮转算法）
                    self.handle_uncompleted_process(next_process)
                    self.current_process = None
            else:
                # 如果无法选择进程，时间向前推进1个单位
                self.current_time += 1

        # 计算整体性能指标
        return self.calculate_system_metrics()

    def get_execution_time(self, process: Process) -> int:
        """获取进程应执行的时间单位，默认是剩余时间"""
        return process.remaining_time

    def handle_uncompleted_process(self, process: Process) -> None:
        """处理未完成的进程，默认实现是放回就绪队列"""
        if process.remaining_time > 0 and process not in self.ready_queue:
            self.ready_queue.append(process)

    def calculate_system_metrics(self) -> Dict:
        """计算系统整体性能指标"""
        # 确保所有进程都已计算指标
        for process in self.completed_processes:
            process.calculate_metrics()
        
        # 计算总体指标
        if not self.completed_processes:
            return {
                "cpu_utilization": 0,
                "throughput": 0,
                "avg_turnaround_time": 0,
                "avg_waiting_time": 0,
                "avg_response_time": 0
            }
        
        # 总运行时间
        total_time = max([p.completion_time for p in self.completed_processes], default=0)
        if total_time == 0:
            return {
                "cpu_utilization": 0,
                "throughput": 0,
                "avg_turnaround_time": 0,
                "avg_waiting_time": 0,
                "avg_response_time": 0
            }
        
        # 计算实际CPU执行时间
        # 通过统计execution_sequence中的所有时间段
        cpu_burst_time = sum([ex["end_time"] - ex["start_time"] for ex in self.execution_sequence 
                             if ex["status"] != "抢占"])
        
        # CPU利用率 = CPU执行时间 / 总时间
        cpu_utilization = cpu_burst_time / total_time
        
        # 吞吐量 = 完成进程数 / 总时间
        throughput = len(self.completed_processes) / total_time
        
        # 平均周转时间
        avg_turnaround_time = sum([p.turnaround_time for p in self.completed_processes]) / len(self.completed_processes)
        
        # 平均等待时间
        avg_waiting_time = sum([p.waiting_time for p in self.completed_processes]) / len(self.completed_processes)
        
        # 平均响应时间
        avg_response_time = sum([p.response_time for p in self.completed_processes]) / len(self.completed_processes)
        
        return {
            "cpu_utilization": cpu_utilization,
            "throughput": throughput,
            "avg_turnaround_time": avg_turnaround_time,
            "avg_waiting_time": avg_waiting_time,
            "avg_response_time": avg_response_time
        }

    def print_results(self) -> None:
        """打印调度结果"""
        # 清屏
        clear_screen()
        
        print(f"\n===== {self.name} 调度算法 =====")
        
        # 打印配置参数
        if self.config:
            print("\n配置参数:")
            for key, value in self.config.items():
                print(f"  {key}: {value}")
        
        # 打印所有进程信息
        print("\n进程信息:")
        for process in self.processes:
            print(f"  {process}")
        
        # 打印执行序列
        print("\n执行序列:")
        for ex in self.execution_sequence:
            print(f"  时间 [{ex['start_time']}-{ex['end_time']}]: 进程 {ex['pid']} ({ex['status']})")
        
        # 打印每个进程的性能指标
        print("\n进程性能指标:")
        headers = ["进程ID", "到达时间", "运行时间", "完成时间", "周转时间", "等待时间", "响应时间"]
        print("  " + " | ".join(headers))
        print("  " + "-" * (len(" | ".join(headers)) + 2))
        
        for process in sorted(self.processes, key=lambda p: p.pid):
            if process.completion_time is not None:  # 只显示已完成的进程
                row = [
                    str(process.pid),
                    str(process.arrival_time),
                    str(process.run_time),
                    str(process.completion_time),
                    str(process.turnaround_time),
                    str(process.waiting_time),
                    str(process.response_time)
                ]
                print("  " + " | ".join(row))
        
        # 打印系统整体性能指标
        metrics = self.calculate_system_metrics()
        print("\n系统性能指标:")
        print(f"  CPU 利用率: {metrics['cpu_utilization']*100:.2f}%")
        print(f"  吞吐量: {metrics['throughput']:.2f} 进程/时间单位")
        print(f"  平均周转时间: {metrics['avg_turnaround_time']:.2f} 时间单位")
        print(f"  平均等待时间: {metrics['avg_waiting_time']:.2f} 时间单位")
        print(f"  平均响应时间: {metrics['avg_response_time']:.2f} 时间单位")

    def visualize(self) -> None:
        """使用Plotly可视化调度过程和性能指标"""
        # 为每个进程分配颜色
        process_colors = {}
        for i, process in enumerate(self.processes):
            # 使用Plotly默认颜色方案
            color_index = i % 10
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            process_colors[process.pid] = colors[color_index]
        
        # 创建子图：甘特图和性能指标表格
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            specs=[[{"type": "scatter"}], [{"type": "table"}]],
            subplot_titles=(f"{self.name} 调度甘特图", "系统性能指标")
        )
        
        # 添加甘特图
        for ex in self.execution_sequence:
            if ex["status"] != "抢占":  # 不显示抢占事件
                fig.add_trace(
                    go.Bar(
                        x=[ex["end_time"] - ex["start_time"]],  # 宽度
                        y=[f"进程 {ex['pid']}"],  # 进程标识
                        orientation='h',  # 水平条形图
                        base=ex["start_time"],  # 起始位置
                        marker=dict(color=process_colors[ex["pid"]]),
                        text=f"{ex['start_time']}-{ex['end_time']}",  # 显示时间范围
                        hoverinfo="text",
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 性能指标表格
        metrics = self.calculate_system_metrics()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['指标', '值'],
                    font=dict(size=14, family='SimHei'),
                    align='center'
                ),
                cells=dict(
                    values=[
                        ['CPU 利用率', '吞吐量', '平均周转时间', '平均等待时间', '平均响应时间'],
                        [
                            f"{metrics['cpu_utilization']*100:.2f}%",
                            f"{metrics['throughput']:.2f} 进程/时间单位",
                            f"{metrics['avg_turnaround_time']:.2f} 时间单位",
                            f"{metrics['avg_waiting_time']:.2f} 时间单位",
                            f"{metrics['avg_response_time']:.2f} 时间单位"
                        ]
                    ],
                    font=dict(size=12, family='SimHei'),
                    align='center'
                )
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title=f"{self.name} 调度算法可视化",
            title_font=dict(size=20, family='SimHei'),
            barmode='stack',
            yaxis=dict(
                title='进程',
                titlefont=dict(size=14, family='SimHei'),
                autorange="reversed"  # 进程ID从上到下排列
            ),
            xaxis=dict(
                title='时间',
                titlefont=dict(size=14, family='SimHei'),
                dtick=1,  # 时间刻度为1
                showgrid=True
            ),
            height=800,
            font=dict(family='SimHei')  # 使用黑体支持中文
        )
        
        fig.show()


class FCFSScheduler(Scheduler):
    """先来先服务调度器(FCFS)"""
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "先来先服务(FCFS)")

    def select_process(self) -> Optional[Process]:
        """选择到达时间最早的进程"""
        if not self.ready_queue:
            return None
        
        # 按到达时间排序，如果到达时间相同则按进程ID排序
        return sorted(self.ready_queue, key=lambda p: (p.arrival_time, p.pid))[0]


class NonPreemptiveSJFScheduler(Scheduler):
    """非抢占式短作业优先调度器(SJF)"""
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "非抢占式短作业优先(SJF)")

    def select_process(self) -> Optional[Process]:
        """选择运行时间最短的进程"""
        if not self.ready_queue:
            return None
        
        # 如果当前有进程在执行且未完成，继续执行该进程
        if self.current_process and self.current_process.remaining_time > 0:
            return self.current_process
        
        # 否则，选择运行时间最短的进程
        return sorted(self.ready_queue, key=lambda p: (p.run_time, p.arrival_time))[0]


class PreemptiveSJFScheduler(Scheduler):
    """抢占式短作业优先调度器(SRTF)"""
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "抢占式短作业优先(SRTF)")

    def is_preemptive(self) -> bool:
        return True

    def select_process(self) -> Optional[Process]:
        """选择剩余运行时间最短的进程"""
        if not self.ready_queue and (not self.current_process or self.current_process.remaining_time <= 0):
            return None
        
        # 将当前进程加入考虑范围
        candidates = self.ready_queue.copy()
        if self.current_process and self.current_process.remaining_time > 0:
            candidates.append(self.current_process)
        
        # 选择剩余运行时间最短的进程
        return sorted(candidates, key=lambda p: (p.remaining_time, p.arrival_time))[0]

    def get_execution_time(self, process: Process) -> int:
        """获取执行时间，直到下一个进程到达或当前进程完成"""
        if not self.processes:
            return process.remaining_time
        
        # 找出下一个将到达的进程
        next_arrivals = [p.arrival_time for p in self.processes 
                          if p.arrival_time > self.current_time and 
                          p.arrival_time < self.current_time + process.remaining_time]
        
        if next_arrivals:
            next_arrival_time = min(next_arrivals)
            return next_arrival_time - self.current_time
        else:
            return process.remaining_time


class NonPreemptivePriorityScheduler(Scheduler):
    """非抢占式优先级调度器"""
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "非抢占式优先级调度")
        
        # 检查是否所有进程都有优先级
        self._check_priorities()

    def _check_priorities(self) -> None:
        """检查是否所有进程都有优先级"""
        for process in self.processes:
            if process.priority is None:
                raise ValueError(f"进程 {process.pid} 没有定义优先级")

    def select_process(self) -> Optional[Process]:
        """选择优先级最高的进程（数值最小）"""
        if not self.ready_queue:
            return None
        
        # 如果当前有进程在执行且未完成，继续执行该进程
        if self.current_process and self.current_process.remaining_time > 0:
            return self.current_process
        
        # 否则，选择优先级最高的进程
        return sorted(self.ready_queue, key=lambda p: (p.priority, p.arrival_time))[0]


class PreemptivePriorityScheduler(Scheduler):
    """抢占式优先级调度器"""
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "抢占式优先级调度")
        
        # 检查是否所有进程都有优先级
        self._check_priorities()

    def _check_priorities(self) -> None:
        """检查是否所有进程都有优先级"""
        for process in self.processes:
            if process.priority is None:
                raise ValueError(f"进程 {process.pid} 没有定义优先级")

    def is_preemptive(self) -> bool:
        return True

    def select_process(self) -> Optional[Process]:
        """选择优先级最高的进程（数值最小）"""
        if not self.ready_queue and (not self.current_process or self.current_process.remaining_time <= 0):
            return None
        
        # 将当前进程加入考虑范围
        candidates = self.ready_queue.copy()
        if self.current_process and self.current_process.remaining_time > 0:
            candidates.append(self.current_process)
        
        # 选择优先级最高的进程
        return sorted(candidates, key=lambda p: (p.priority, p.arrival_time))[0]

    def get_execution_time(self, process: Process) -> int:
        """获取执行时间，直到下一个进程到达或当前进程完成"""
        if not self.processes:
            return process.remaining_time
        
        # 找出下一个将到达的进程
        next_arrivals = [p.arrival_time for p in self.processes 
                          if p.arrival_time > self.current_time and 
                          p.arrival_time < self.current_time + process.remaining_time]
        
        if next_arrivals:
            next_arrival_time = min(next_arrivals)
            return next_arrival_time - self.current_time
        else:
            return process.remaining_time


class RoundRobinScheduler(Scheduler):
    """时间片轮转调度器(RR)"""
    def __init__(self, processes: List[Process], time_quantum: int):
        super().__init__(processes, "时间片轮转(RR)")
        self.time_quantum = time_quantum
        self.config = {"时间片大小": time_quantum}

    def is_preemptive(self) -> bool:
        return True

    def select_process(self) -> Optional[Process]:
        """按FIFO顺序选择进程"""
        if not self.ready_queue:
            return None
        
        # 始终选择队首进程
        return self.ready_queue[0]

    def get_execution_time(self, process: Process) -> int:
        """执行时间为时间片和剩余时间的较小值"""
        return min(self.time_quantum, process.remaining_time)

    def handle_uncompleted_process(self, process: Process) -> None:
        """将未完成的进程放到就绪队列尾部"""
        if process.remaining_time > 0:
            self.ready_queue.append(process)


class MultiLevelQueueScheduler(Scheduler):
    """多级队列调度器"""
    def __init__(self, processes: List[Process], queue_configs: List[Dict]):
        """
        初始化多级队列调度器
        queue_configs: 队列配置列表，形如 [{"algorithm": "RR", "time_quantum": 2}, {"algorithm": "FCFS"}]
        """
        super().__init__(processes, "多级队列调度")
        
        self.queue_configs = queue_configs
        self.queues = [[] for _ in range(len(queue_configs))]  # 多个就绪队列
        
        # 设置每个进程的初始队列
        for process in self.processes:
            if process.priority is None:
                raise ValueError(f"进程 {process.pid} 没有定义优先级")
            
            # 根据优先级分配到队列
            queue_idx = min(process.priority, len(self.queues) - 1)
            process.current_queue = queue_idx
        
        # 配置信息
        config_str = ""
        for i, cfg in enumerate(queue_configs):
            if cfg["algorithm"] == "RR":
                config_str += f"队列{i}:{cfg['algorithm']}(时间片:{cfg['time_quantum']}) "
            else:
                config_str += f"队列{i}:{cfg['algorithm']} "
        
        self.config = {
            "队列数量": len(queue_configs),
            "队列配置": config_str
        }

    def update_ready_queue(self) -> None:
        """更新多级就绪队列"""
        for process in self.processes:
            if (process.arrival_time <= self.current_time and 
                process.remaining_time > 0 and 
                process not in self.completed_processes and
                process != self.current_process and
                not any(process in q for q in self.queues)):
                
                queue_idx = process.current_queue
                self.queues[queue_idx].append(process)
        
        # 更新主就绪队列（仅用于兼容基类）
        self.ready_queue = []
        for q in self.queues:
            self.ready_queue.extend(q)

    def select_process(self) -> Optional[Process]:
        """从高优先级队列开始选择进程"""
        # 检查每个队列，从高优先级到低优先级
        for i, queue in enumerate(self.queues):
            if queue:
                # 根据队列的调度算法选择进程
                algorithm = self.queue_configs[i]["algorithm"]
                
                if algorithm == "FCFS":
                    # 选择到达时间最早的进程
                    return sorted(queue, key=lambda p: (p.arrival_time, p.pid))[0]
                elif algorithm == "SJF":
                    # 选择运行时间最短的进程
                    return sorted(queue, key=lambda p: (p.run_time, p.arrival_time))[0]
                elif algorithm == "Priority":
                    # 选择优先级最高的进程
                    return sorted(queue, key=lambda p: (p.priority, p.arrival_time))[0]
                elif algorithm == "RR":
                    # 选择队首进程
                    return queue[0]
        
        return None

    def is_preemptive(self) -> bool:
        """多级队列调度的抢占性取决于当前进程所在队列的调度算法"""
        if self.current_process:
            queue_idx = self.current_process.current_queue
            algorithm = self.queue_configs[queue_idx]["algorithm"]
            return algorithm in ["RR", "SRTF", "Priority"]
        return False

    def get_execution_time(self, process: Process) -> int:
        """获取执行时间，取决于进程所在队列的调度算法"""
        queue_idx = process.current_queue
        algorithm = self.queue_configs[queue_idx]["algorithm"]
        
        if algorithm == "RR":
            time_quantum = self.queue_configs[queue_idx]["time_quantum"]
            return min(time_quantum, process.remaining_time)
        else:
            # 对于非抢占式算法或者抢占式优先级算法
            # 计算到下一个高优先级进程到达前的时间
            next_arrival_time = float('inf')
            
            # 检查高优先级队列的进程到达时间
            for i in range(queue_idx):
                for p in self.processes:
                    if (p.current_queue == i and 
                        p.arrival_time > self.current_time and 
                        p.arrival_time < self.current_time + process.remaining_time):
                        next_arrival_time = min(next_arrival_time, p.arrival_time)
            
            if next_arrival_time != float('inf'):
                return next_arrival_time - self.current_time
            else:
                return process.remaining_time

    def handle_uncompleted_process(self, process: Process) -> None:
        """处理未完成的进程，根据算法放回相应队列"""
        if process.remaining_time > 0:
            queue_idx = process.current_queue
            algorithm = self.queue_configs[queue_idx]["algorithm"]
            
            if algorithm == "RR":
                # 将进程放回队列尾部
                self.queues[queue_idx].append(process)
            else:
                # 对于其他算法，也放回队列
                self.queues[queue_idx].append(process)


class MultilevelFeedbackQueueScheduler(Scheduler):
    """多级反馈队列调度器"""
    def __init__(self, processes: List[Process], queue_count: int, time_quantums: List[int], enable_aging: bool = False, aging_threshold: int = 10):
        super().__init__(processes, "多级反馈队列调度")
        
        self.queue_count = queue_count
        self.time_quantums = time_quantums
        self.queues = [[] for _ in range(queue_count)]  # 多个就绪队列
        self.enable_aging = enable_aging
        self.aging_threshold = aging_threshold
        self.process_waiting_time = {process.pid: 0 for process in processes}  # 跟踪每个进程在低优先级队列的等待时间
        
        # 所有进程初始时在最高优先级队列
        for process in self.processes:
            process.current_queue = 0
        
        # 配置信息
        time_quantums_str = ", ".join(str(tq) for tq in time_quantums)
        self.config = {
            "队列数量": queue_count,
            "各队列时间片": time_quantums_str,
            "启用老化": "是" if enable_aging else "否"
        }
        if enable_aging:
            self.config["老化阈值"] = aging_threshold

    def update_ready_queue(self) -> None:
        """更新多级反馈就绪队列"""
        for process in self.processes:
            if (process.arrival_time <= self.current_time and 
                process.remaining_time > 0 and 
                process not in self.completed_processes and
                process != self.current_process and
                not any(process in q for q in self.queues)):
                
                # 新进程进入最高优先级队列
                if process.start_time is None:
                    process.current_queue = 0
                
                queue_idx = process.current_queue
                self.queues[queue_idx].append(process)
        
        # 老化机制
        if self.enable_aging:
            self._apply_aging()
        
        # 更新主就绪队列（仅用于兼容基类）
        self.ready_queue = []
        for q in self.queues:
            self.ready_queue.extend(q)

    def _apply_aging(self) -> None:
        """应用老化机制：将在低优先级队列等待时间过长的进程移回较高优先级队列"""
        for i in range(1, self.queue_count):  # 从第二个队列开始
            for process in self.queues[i][:]:  # 使用副本迭代
                self.process_waiting_time[process.pid] += 1
                if self.process_waiting_time[process.pid] >= self.aging_threshold:
                    # 将进程提升一个优先级
                    new_queue = max(0, process.current_queue - 1)
                    if new_queue != process.current_queue:
                        self.queues[i].remove(process)
                        process.current_queue = new_queue
                        self.queues[new_queue].append(process)
                        self.process_waiting_time[process.pid] = 0  # 重置等待时间

    def select_process(self) -> Optional[Process]:
        """从高优先级队列开始选择进程"""
        # 检查每个队列，从高优先级到低优先级
        for i, queue in enumerate(self.queues):
            if queue:
                # 最后一个队列使用FCFS
                if i == self.queue_count - 1:
                    return sorted(queue, key=lambda p: (p.arrival_time, p.pid))[0]
                else:
                    # 其他队列使用RR
                    return queue[0]
        
        return None

    def is_preemptive(self) -> bool:
        """多级反馈队列是抢占式的"""
        return True

    def get_execution_time(self, process: Process) -> int:
        """获取执行时间，根据进程所在队列决定时间片大小"""
        queue_idx = process.current_queue
        
        # 最后一个队列使用FCFS
        if queue_idx == self.queue_count - 1:
            return process.remaining_time
        else:
            # 其他队列使用RR，时间片大小按队列不同
            time_quantum = self.time_quantums[queue_idx]
            return min(time_quantum, process.remaining_time)

    def handle_uncompleted_process(self, process: Process) -> None:
        """处理未完成的进程，如果在当前队列的时间片用完，则降级到下一级队列"""
        if process.remaining_time > 0:
            current_queue = process.current_queue
            
            # 检查是否用完时间片（对于非最后一级队列）
            if current_queue < self.queue_count - 1:
                time_quantum = self.time_quantums[current_queue]
                
                # 如果执行了完整的时间片且未完成，则降级
                executed_time = self.time_quantums[current_queue] - process.remaining_time
                if executed_time >= time_quantum:
                    # 降级到下一级队列
                    next_queue = min(current_queue + 1, self.queue_count - 1)
                    process.current_queue = next_queue
                    self.queues[next_queue].append(process)
                    # 重置该进程的等待时间
                    self.process_waiting_time[process.pid] = 0
                else:
                    # 未用完时间片，仍留在当前队列
                    self.queues[current_queue].append(process)
            else:
                # 最后一级队列使用FCFS，进程仍然留在该队列
                self.queues[current_queue].append(process)
            
            # 更新其他进程的等待时间
            for pid in self.process_waiting_time:
                if pid != process.pid:
                    self.process_waiting_time[pid] += 1


# 辅助函数

def clear_screen() -> None:
    """清屏函数，根据操作系统选择不同的清屏命令"""
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')


def validate_processes(processes: List[Process], algorithm: str) -> bool:
    """验证进程数据是否适合选定的调度算法"""
    if algorithm in ["非抢占式优先级调度", "抢占式优先级调度", "多级队列调度", "多级反馈队列调度"]:
        for process in processes:
            if process.priority is None:
                return False
    return True


def get_integer_input(prompt: str, min_value: int = 0, max_value: int = float('inf')) -> int:
    """获取整数输入，带有范围验证"""
    while True:
        try:
            value = int(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"请输入范围在 {min_value} 到 {max_value} 之间的整数。")
        except ValueError:
            print("无效输入，请输入一个整数。")


def get_yes_no_input(prompt: str) -> bool:
    """获取是/否输入"""
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes', '是', '1']:
            return True
        elif response in ['n', 'no', '否', '0']:
            return False
        else:
            print("无效输入，请输入 是/否 (y/n)。")


def manual_input_processes() -> List[Process]:
    """手动输入进程数据"""
    processes = []
    
    process_count = get_integer_input("请输入进程数量: ", 1, 100)
    
    need_priority = get_yes_no_input("是否需要为进程指定优先级? (y/n): ")
    
    for i in range(process_count):
        pid = i + 1
        arrival_time = get_integer_input(f"请输入进程 {pid} 的到达时间: ", 0)
        run_time = get_integer_input(f"请输入进程 {pid} 的运行时间: ", 1)
        
        priority = None
        if need_priority:
            priority = get_integer_input(f"请输入进程 {pid} 的优先级 (较小值优先级更高): ", 0)
        
        processes.append(Process(pid, arrival_time, run_time, priority))
    
    return processes


def random_generate_processes() -> List[Process]:
    """随机生成进程数据"""
    processes = []
    
    process_count = get_integer_input("请输入要生成的进程数量: ", 1, 100)
    max_arrival_time = get_integer_input("请输入最大到达时间: ", 0)
    max_run_time = get_integer_input("请输入最大运行时间: ", 1)
    
    need_priority = get_yes_no_input("是否需要为进程生成优先级? (y/n): ")
    
    for i in range(process_count):
        pid = i + 1
        arrival_time = random.randint(0, max_arrival_time)
        run_time = random.randint(1, max_run_time)
        
        priority = None
        if need_priority:
            priority = random.randint(0, 9)
        
        processes.append(Process(pid, arrival_time, run_time, priority))
    
    return processes


def run_scheduling_simulation():
    """运行调度模拟的主函数"""
    # 清屏并显示欢迎信息
    clear_screen()
    print("\n====== 操作系统调度算法可视化工具 ======\n")
    
    # 1. 进程数据输入方式选择
    print("请选择进程数据输入方式:")
    print("1. 手动输入")
    print("2. 随机生成")
    
    input_choice = get_integer_input("请输入选择 (1/2): ", 1, 2)
    
    # 根据选择获取进程数据
    if input_choice == 1:
        processes = manual_input_processes()
    else:
        processes = random_generate_processes()
    
    # 2. 选择调度算法
    clear_screen()
    print("\n请选择要模拟的调度算法:")
    print("1. 先来先服务 (FCFS)")
    print("2. 非抢占式短作业优先 (SJF)")
    print("3. 抢占式短作业优先 (SRTF)")
    print("4. 非抢占式优先级调度")
    print("5. 抢占式优先级调度")
    print("6. 时间片轮转 (RR)")
    print("7. 多级队列调度")
    print("8. 多级反馈队列调度")
    
    algorithm_choice = get_integer_input("请输入选择 (1-8): ", 1, 8)
    
    # 3. 根据算法选择，可能需要额外的参数配置
    scheduler = None
    
    if algorithm_choice == 1:  # FCFS
        scheduler = FCFSScheduler(processes)
    
    elif algorithm_choice == 2:  # 非抢占式SJF
        scheduler = NonPreemptiveSJFScheduler(processes)
    
    elif algorithm_choice == 3:  # 抢占式SJF (SRTF)
        scheduler = PreemptiveSJFScheduler(processes)
    
    elif algorithm_choice == 4:  # 非抢占式优先级调度
        if not validate_processes(processes, "非抢占式优先级调度"):
            print("错误: 执行优先级调度算法需要为所有进程指定优先级!")
            return
        scheduler = NonPreemptivePriorityScheduler(processes)
    
    elif algorithm_choice == 5:  # 抢占式优先级调度
        if not validate_processes(processes, "抢占式优先级调度"):
            print("错误: 执行优先级调度算法需要为所有进程指定优先级!")
            return
        scheduler = PreemptivePriorityScheduler(processes)
    
    elif algorithm_choice == 6:  # 时间片轮转 (RR)
        time_quantum = get_integer_input("请输入时间片大小: ", 1)
        scheduler = RoundRobinScheduler(processes, time_quantum)
    
    elif algorithm_choice == 7:  # 多级队列调度
        if not validate_processes(processes, "多级队列调度"):
            print("错误: 执行多级队列调度算法需要为所有进程指定优先级!")
            return
        
        queue_count = get_integer_input("请输入队列数量: ", 2, 5)
        queue_configs = []
        
        for i in range(queue_count):
            print(f"\n配置队列 {i}:")
            print("可选调度算法:")
            print("1. 先来先服务 (FCFS)")
            print("2. 短作业优先 (SJF)")
            print("3. 优先级调度 (Priority)")
            print("4. 时间片轮转 (RR)")
            
            alg_choice = get_integer_input("请为该队列选择调度算法 (1-4): ", 1, 4)
            
            config = {}
            if alg_choice == 1:
                config["algorithm"] = "FCFS"
            elif alg_choice == 2:
                config["algorithm"] = "SJF"
            elif alg_choice == 3:
                config["algorithm"] = "Priority"
            elif alg_choice == 4:
                config["algorithm"] = "RR"
                time_quantum = get_integer_input("请输入时间片大小: ", 1)
                config["time_quantum"] = time_quantum
            
            queue_configs.append(config)
        
        scheduler = MultiLevelQueueScheduler(processes, queue_configs)
    
    elif algorithm_choice == 8:  # 多级反馈队列调度
        queue_count = get_integer_input("请输入队列数量: ", 2, 5)
        time_quantums = []
        
        for i in range(queue_count - 1):  # 最后一个队列使用FCFS
            time_quantum = get_integer_input(f"请输入队列 {i} 的时间片大小: ", 1)
            time_quantums.append(time_quantum)
        
        time_quantums.append(0)  # 最后一个队列使用FCFS，时间片设为0表示不限
        
        enable_aging = get_yes_no_input("是否启用老化机制? (y/n): ")
        aging_threshold = 10  # 默认值
        
        if enable_aging:
            aging_threshold = get_integer_input("请输入老化阈值 (在低优先级队列等待多少时间单位后提升): ", 1)
        
        scheduler = MultilevelFeedbackQueueScheduler(processes, queue_count, time_quantums, enable_aging, aging_threshold)
    
    # 4. 运行调度算法
    if scheduler:
        metrics = scheduler.run()
        
        # 5. 显示结果
        scheduler.print_results()
        
        # 是否显示可视化结果
        if get_yes_no_input("\n是否显示可视化结果? (y/n): "):
            scheduler.visualize()


if __name__ == "__main__":
    run_scheduling_simulation()
