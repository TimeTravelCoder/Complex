import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple, Union

class Process:
    """表示一个进程，包含进程的各种属性和指标"""
    
    def __init__(self, pid: str, arrival_time: int, run_time: int, priority: Optional[int] = None):
        self.pid = pid  # 进程ID
        self.arrival_time = arrival_time  # 到达时间
        self.run_time = run_time  # 运行时间
        self.priority = priority  # 优先级（数值越小优先级越高）
        self.remaining_time = run_time  # 剩余运行时间
        self.start_time = None  # 首次执行时间
        self.completion_time = None  # 完成时间
        self.waiting_time = 0  # 等待时间
        self.turnaround_time = 0  # 周转时间
        self.response_time = None  # 响应时间

    def __repr__(self) -> str:
        return f"Process(pid={self.pid}, arrival_time={self.arrival_time}, run_time={self.run_time})"

class Scheduler:
    """调度器基类，定义了调度器的通用接口和方法"""
    
    def __init__(self, processes: List[Process]):
        self.processes = processes.copy()  # 进程列表
        self.ready_queue: List[Process] = []  # 就绪队列
        self.current_time = 0  # 当前时间
        self.running_process: Optional[Process] = None  # 当前运行的进程
        self.execution_history: List[Dict[str, Union[str, int]]] = []  # 执行历史
        self.finished_processes: List[Process] = []  # 已完成的进程
        self.time_quantum = None  # 时间片（用于RR算法）
        self.context_switch = False  # 是否发生了上下文切换
        
    def update_ready_queue(self) -> None:
        """将已到达且未完成的进程加入就绪队列"""
        for process in self.processes:
            if (process.arrival_time <= self.current_time and 
                process not in self.ready_queue and 
                process not in self.finished_processes and
                process is not self.running_process):
                self.ready_queue.append(process)
    
    def select_process(self) -> Optional[Process]:
        """选择下一个要执行的进程（由子类实现）"""
        raise NotImplementedError("子类必须实现select_process方法")
    
    def execute(self, time_units: int) -> None:
        """执行当前选中的进程指定的时间单位"""
        if not self.running_process:
            return
            
        start_time = self.current_time
        execution_time = min(time_units, self.running_process.remaining_time)
        
        # 更新首次执行时间（响应时间）
        if self.running_process.start_time is None:
            self.running_process.start_time = self.current_time
            self.running_process.response_time = self.current_time - self.running_process.arrival_time
        
        # 更新剩余时间
        self.running_process.remaining_time -= execution_time
        self.current_time += execution_time
        
        # 记录执行历史
        self.execution_history.append({
            'pid': self.running_process.pid,
            'start_time': start_time,
            'end_time': self.current_time,
            'action': '执行'
        })
        
        # 检查进程是否完成
        if self.running_process.remaining_time == 0:
            self.running_process.completion_time = self.current_time
            self.running_process.turnaround_time = (
                self.running_process.completion_time - self.running_process.arrival_time
            )
            self.running_process.waiting_time = (
                self.running_process.turnaround_time - self.running_process.run_time
            )
            self.finished_processes.append(self.running_process)
            self.running_process = None
            self.context_switch = True
    
    def run(self) -> None:
        """运行调度算法直到所有进程完成"""
        while len(self.finished_processes) < len(self.processes):
            self.update_ready_queue()
            
            # 如果没有运行的进程，选择一个新进程
            if not self.running_process and self.ready_queue:
                self.running_process = self.select_process()
                self.ready_queue.remove(self.running_process)
                self.context_switch = True
            
            # 执行一个时间单位
            if self.running_process:
                self.execute(1)
            else:
                # CPU空闲
                self.current_time += 1
                self.execution_history.append({
                    'pid': '空闲',
                    'start_time': self.current_time - 1,
                    'end_time': self.current_time,
                    'action': '空闲'
                })
    
    def get_metrics(self) -> Tuple[Dict[str, float], List[Dict[str, Union[str, int]]]]:
        """计算并返回性能指标"""
        if not self.finished_processes:
            return {}, []
            
        # 计算系统整体性能指标
        total_time = self.current_time
        cpu_utilization = sum(p.run_time for p in self.finished_processes) / total_time
        throughput = len(self.finished_processes) / total_time
        avg_turnaround_time = sum(p.turnaround_time for p in self.finished_processes) / len(self.finished_processes)
        avg_waiting_time = sum(p.waiting_time for p in self.finished_processes) / len(self.finished_processes)
        avg_response_time = sum(p.response_time for p in self.finished_processes) / len(self.finished_processes)
        
        system_metrics = {
            'CPU利用率': cpu_utilization,
            '吞吐量': throughput,
            '平均周转时间': avg_turnaround_time,
            '平均等待时间': avg_waiting_time,
            '平均响应时间': avg_response_time
        }
        
        # 计算每个进程的性能指标
        process_metrics = []
        for p in sorted(self.finished_processes, key=lambda x: x.arrival_time):
            process_metrics.append({
                '进程ID': p.pid,
                '到达时间': p.arrival_time,
                '运行时间': p.run_time,
                '完成时间': p.completion_time,
                '周转时间': p.turnaround_time,
                '等待时间': p.waiting_time,
                '响应时间': p.response_time
            })
        
        return system_metrics, process_metrics

# 具体调度算法实现
class FCFS(Scheduler):
    """先来先服务(FCFS)调度算法"""
    
    def select_process(self) -> Optional[Process]:
        """选择到达时间最早的进程，如果到达时间相同则选择PID较小的进程"""
        if not self.ready_queue:
            return None
        return min(self.ready_queue, key=lambda p: (p.arrival_time, p.pid))

class NonPreemptiveSJF(Scheduler):
    """非抢占式短作业优先(SJF)调度算法"""
    
    def select_process(self) -> Optional[Process]:
        """选择运行时间最短的进程，如果运行时间相同则选择到达时间最早的进程"""
        if not self.ready_queue:
            return None
        return min(self.ready_queue, key=lambda p: (p.run_time, p.arrival_time, p.pid))

class PreemptiveSJF(Scheduler):
    """抢占式短作业优先(SRTF)调度算法"""
    
    def update_ready_queue(self) -> None:
        """重写更新就绪队列的方法，处理抢占情况"""
        super().update_ready_queue()
        
        # 检查是否需要抢占当前运行的进程
        if (self.running_process and self.ready_queue and 
            any(p.remaining_time < self.running_process.remaining_time for p in self.ready_queue)):
            
            # 记录抢占事件
            self.execution_history.append({
                'pid': self.running_process.pid,
                'start_time': self.current_time,
                'end_time': self.current_time,
                'action': '被抢占'
            })
            
            # 将当前运行的进程放回就绪队列
            self.ready_queue.append(self.running_process)
            self.context_switch = True
            
            # 选择剩余时间最短的进程
            self.running_process = min(self.ready_queue, key=lambda p: p.remaining_time)
            self.ready_queue.remove(self.running_process)
    
    def select_process(self) -> Optional[Process]:
        """选择剩余时间最短的进程"""
        if not self.ready_queue:
            return None
        return min(self.ready_queue, key=lambda p: p.remaining_time)

class NonPreemptivePriority(Scheduler):
    """非抢占式优先级调度算法"""
    
    def __init__(self, processes: List[Process]):
        super().__init__(processes)
        # 检查所有进程是否都有优先级
        for p in processes:
            if p.priority is None:
                raise ValueError("非抢占式优先级调度要求所有进程都有优先级")
    
    def select_process(self) -> Optional[Process]:
        """选择优先级最高的进程（优先级数值最小）"""
        if not self.ready_queue:
            return None
        return min(self.ready_queue, key=lambda p: (p.priority, p.arrival_time, p.pid))

class PreemptivePriority(Scheduler):
    """抢占式优先级调度算法"""
    
    def __init__(self, processes: List[Process]):
        super().__init__(processes)
        # 检查所有进程是否都有优先级
        for p in processes:
            if p.priority is None:
                raise ValueError("抢占式优先级调度要求所有进程都有优先级")
    
    def update_ready_queue(self) -> None:
        """重写更新就绪队列的方法，处理抢占情况"""
        super().update_ready_queue()
        
        # 检查是否需要抢占当前运行的进程
        if (self.running_process and self.ready_queue and 
            any(p.priority < self.running_process.priority for p in self.ready_queue)):
            
            # 记录抢占事件
            self.execution_history.append({
                'pid': self.running_process.pid,
                'start_time': self.current_time,
                'end_time': self.current_time,
                'action': '被抢占'
            })
            
            # 将当前运行的进程放回就绪队列
            self.ready_queue.append(self.running_process)
            self.context_switch = True
            
            # 选择优先级最高的进程
            self.running_process = min(self.ready_queue, key=lambda p: p.priority)
            self.ready_queue.remove(self.running_process)
    
    def select_process(self) -> Optional[Process]:
        """选择优先级最高的进程（优先级数值最小）"""
        if not self.ready_queue:
            return None
        return min(self.ready_queue, key=lambda p: p.priority)

class RoundRobin(Scheduler):
    """时间片轮转(RR)调度算法"""
    
    def __init__(self, processes: List[Process], time_quantum: int):
        super().__init__(processes)
        self.time_quantum = time_quantum  # 时间片大小
        self.time_slice_counter = 0  # 当前时间片计数器
    
    def update_ready_queue(self) -> None:
        """重写更新就绪队列的方法，处理时间片用完的情况"""
        super().update_ready_queue()
        
        # 检查时间片是否用完
        if (self.running_process and self.time_slice_counter >= self.time_quantum and 
            self.running_process.remaining_time > 0):
            
            # 记录时间片用完事件
            self.execution_history.append({
                'pid': self.running_process.pid,
                'start_time': self.current_time,
                'end_time': self.current_time,
                'action': '时间片用完'
            })
            
            # 将当前运行的进程放回就绪队列尾部
            self.ready_queue.append(self.running_process)
            self.running_process = None
            self.time_slice_counter = 0
            self.context_switch = True
    
    def select_process(self) -> Optional[Process]:
        """选择就绪队列头部的进程"""
        if not self.ready_queue:
            return None
        self.time_slice_counter = 0  # 重置时间片计数器
        return self.ready_queue.pop(0)
    
    def execute(self, time_units: int) -> None:
        """执行当前选中的进程，考虑时间片限制"""
        if not self.running_process:
            return
            
        execution_time = min(time_units, self.running_process.remaining_time, 
                            self.time_quantum - self.time_slice_counter)
        
        super().execute(execution_time)
        self.time_slice_counter += execution_time

class MultilevelQueue(Scheduler):
    """多级队列调度算法"""
    
    def __init__(self, processes: List[Process], queue_config: List[Dict[str, Union[str, int]]]):
        """
        初始化多级队列调度器
        
        参数:
            processes: 进程列表
            queue_config: 队列配置列表，每个队列配置包含:
                - 'algorithm': 调度算法名称 ('FCFS', 'RR')
                - 'time_quantum': 时间片大小 (仅适用于RR算法)
        """
        super().__init__(processes)
        self.queue_config = queue_config
        self.queues = [[] for _ in range(len(queue_config))]  # 创建多个队列
        
        # 将进程分配到队列
        for process in processes:
            # 简单地根据进程ID的哈希值分配到不同队列
            queue_index = hash(process.pid) % len(self.queues)
            self.queues[queue_index].append(process)
    
    def update_ready_queue(self) -> None:
        """更新就绪队列，优先考虑高优先级队列"""
        super().update_ready_queue()
        
        # 清空就绪队列，重新从各级队列填充
        self.ready_queue = []
        for i, queue in enumerate(self.queues):
            # 只添加当前队列中到达时间小于等于当前时间的进程
            ready_processes = [p for p in queue 
                              if p.arrival_time <= self.current_time 
                              and p not in self.finished_processes 
                              and p is not self.running_process]
            self.ready_queue.extend(ready_processes)
            
            # 如果高优先级队列有进程，则不考虑低优先级队列
            if self.ready_queue:
                break
    
    def select_process(self) -> Optional[Process]:
        """选择下一个要执行的进程，根据队列优先级和调度算法"""
        if not self.ready_queue:
            return None
        
        # 找到第一个有就绪进程的队列
        for i, queue in enumerate(self.queues):
            queue_processes = [p for p in queue if p in self.ready_queue]
            if queue_processes:
                algorithm = self.queue_config[i]['algorithm']
                
                if algorithm == 'FCFS':
                    # 按到达时间排序
                    selected = min(queue_processes, key=lambda p: (p.arrival_time, p.pid))
                elif algorithm == 'RR':
                    # 简单轮转，选择队列头部的进程
                    selected = queue_processes[0]
                else:
                    raise ValueError(f"不支持的调度算法: {algorithm}")
                
                self.ready_queue.remove(selected)
                return selected
        
        return None

class MultilevelFeedbackQueue(Scheduler):
    """多级反馈队列调度算法"""
    
    def __init__(self, processes: List[Process], 
                 queue_config: List[Dict[str, Union[str, int]]],
                 aging_threshold: Optional[int] = None):
        """
        初始化多级反馈队列调度器
        
        参数:
            processes: 进程列表
            queue_config: 队列配置列表，每个队列配置包含:
                - 'time_quantum': 时间片大小
            aging_threshold: 老化阈值，等待时间超过此值的进程将提升优先级
        """
        super().__init__(processes)
        self.queue_config = queue_config
        self.aging_threshold = aging_threshold
        self.queues = [[] for _ in range(len(queue_config))]  # 创建多个队列
        self.time_counters = [0] * len(queue_config)  # 每个队列的时间计数器
        
        # 将所有进程放入最高优先级队列
        for process in processes:
            self.queues[0].append(process)
    
    def update_ready_queue(self) -> None:
        """更新就绪队列，处理老化和时间片用完的情况"""
        super().update_ready_queue()
        
        # 处理老化
        if self.aging_threshold is not None:
            for i in range(1, len(self.queues)):
                for process in self.queues[i]:
                    if process.arrival_time <= self.current_time and process not in self.finished_processes:
                        wait_time = self.current_time - process.arrival_time - sum(
                            h['end_time'] - h['start_time'] 
                            for h in self.execution_history 
                            if h['pid'] == process.pid and h['action'] == '执行'
                        )
                        if wait_time >= self.aging_threshold:
                            # 提升进程优先级
                            self.queues[i].remove(process)
                            self.queues[i-1].append(process)
                            print(f"时间 {self.current_time}: 进程 {process.pid} 因等待时间过长提升优先级")
        
        # 清空就绪队列，重新从各级队列填充
        self.ready_queue = []
        for queue in self.queues:
            # 只添加当前队列中到达时间小于等于当前时间的进程
            ready_processes = [p for p in queue 
                              if p.arrival_time <= self.current_time 
                              and p not in self.finished_processes 
                              and p is not self.running_process]
            self.ready_queue.extend(ready_processes)
            
            # 如果高优先级队列有进程，则不考虑低优先级队列
            if self.ready_queue:
                break
    
    def select_process(self) -> Optional[Process]:
        """选择下一个要执行的进程，根据队列优先级"""
        if not self.ready_queue:
            return None
        
        # 找到第一个有就绪进程的队列
        for i, queue in enumerate(self.queues):
            queue_processes = [p for p in queue if p in self.ready_queue]
            if queue_processes:
                # 选择队列头部的进程
                selected = queue_processes[0]
                self.ready_queue.remove(selected)
                self.current_queue_index = i
                self.time_counters[i] = 0  # 重置时间计数器
                return selected
        
        return None
    
    def execute(self, time_units: int) -> None:
        """执行当前选中的进程，考虑时间片限制和队列优先级"""
        if not self.running_process or self.current_queue_index is None:
            return
            
        current_queue = self.current_queue_index
        time_quantum = self.queue_config[current_queue]['time_quantum']
        
        execution_time = min(time_units, self.running_process.remaining_time, 
                            time_quantum - self.time_counters[current_queue])
        
        super().execute(execution_time)
        self.time_counters[current_queue] += execution_time
        
        # 检查时间片是否用完
        if (self.running_process and self.time_counters[current_queue] >= time_quantum and 
            self.running_process.remaining_time > 0):
            
            # 记录时间片用完事件
            self.execution_history.append({
                'pid': self.running_process.pid,
                'start_time': self.current_time,
                'end_time': self.current_time,
                'action': '时间片用完'
            })
            
            # 如果不是最低优先级队列，将进程移至下一级队列
            if current_queue < len(self.queues) - 1:
                self.queues[current_queue].remove(self.running_process)
                self.queues[current_queue + 1].append(self.running_process)
            
            self.running_process = None
            self.time_counters[current_queue] = 0
            self.context_switch = True

class Visualizer:
    """可视化工具类，用于生成甘特图和性能指标表格"""
    
    @staticmethod
    def visualize(scheduler: Scheduler, algorithm_name: str) -> None:
        """生成甘特图和性能指标表格"""
        system_metrics, process_metrics = scheduler.get_metrics()
        
        # 创建甘特图数据
        tasks = []
        for history in scheduler.execution_history:
            if history['pid'] == '空闲':
                continue
            tasks.append({
                'Task': history['pid'],
                'Start': history['start_time'],
                'Finish': history['end_time'],
                'Resource': history['action']
            })
        
        # 创建甘特图 - 明确指定子图类型
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"{algorithm_name} 调度甘特图", "性能指标"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            specs=[[{"type": "xy"}],  # 第一行是xy类型（用于甘特图）
                   [{"type": "domain"}]]  # 第二行是domain类型（用于表格）
        )
        
        # 添加甘特图
        for task in tasks:
            fig.add_trace(
                go.Bar(
                    x=[task['Finish'] - task['Start']],
                    y=[task['Task']],
                    orientation='h',
                    base=task['Start'],
                    name=task['Resource'],
                    showlegend=(task == tasks[0]),  # 只显示第一个图例
                    hovertemplate=f"进程: {task['Task']}<br>开始时间: {task['Start']}<br>结束时间: {task['Finish']}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # 设置甘特图布局
        fig.update_layout(
            title=f"{algorithm_name} 调度算法可视化",
            xaxis=dict(
                title="时间",
                tickmode='linear',
                dtick=1,
                range=[0, scheduler.current_time]
            ),
            yaxis=dict(
                title="进程",
                categoryorder='array',
                categoryarray=sorted({task['Task'] for task in tasks})
            ),
            bargap=0.1,
            height=800,
            font=dict(family="SimHei, WenQuanYi Micro Hei, Heiti TC")
        )
        
        # 添加性能指标表格
        metric_names = ["CPU利用率", "吞吐量", "平均周转时间", "平均等待时间", "平均响应时间"]
        metric_values = [
            f"{system_metrics['CPU利用率']:.2%}",
            f"{system_metrics['吞吐量']:.2f} 进程/时间单位",
            f"{system_metrics['平均周转时间']:.2f}",
            f"{system_metrics['平均等待时间']:.2f}",
            f"{system_metrics['平均响应时间']:.2f}"
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["性能指标", "数值"],
                    font=dict(size=14, family="SimHei"),
                    align="center"
                ),
                cells=dict(
                    values=[metric_names, metric_values],
                    font=dict(size=12, family="SimHei"),
                    align="center"
                )
            ),
            row=2, col=1
        )
        
        # 显示图表
        fig.show()

def get_valid_input(prompt: str, validation_func, error_msg: str) -> any:
    """获取有效的用户输入"""
    while True:
        user_input = input(prompt)
        try:
            value = validation_func(user_input)
            return value
        except ValueError:
            print(error_msg)

def manual_input_processes() -> List[Process]:
    """手动输入进程信息"""
    processes = []
    print("\n=== 手动输入进程信息 ===")
    print("请输入进程信息（每行一个进程，格式：进程ID,到达时间,运行时间[,优先级]）")
    print("输入空行结束输入")
    
    while True:
        line = input("进程信息 (格式: PID,到达时间,运行时间[,优先级]): ").strip()
        if not line:
            break
            
        parts = line.split(',')
        if len(parts) not in [3, 4]:
            print("错误: 格式不正确，请重新输入")
            continue
            
        try:
            pid = parts[0].strip()
            arrival_time = int(parts[1].strip())
            run_time = int(parts[2].strip())
            priority = int(parts[3].strip()) if len(parts) == 4 else None
            
            processes.append(Process(pid, arrival_time, run_time, priority))
        except ValueError:
            print("错误: 到达时间、运行时间和优先级必须是整数，请重新输入")
    
    return processes

def random_generate_processes() -> List[Process]:
    """随机生成进程信息"""
    print("\n=== 随机生成进程信息 ===")
    num_processes = get_valid_input(
        "请输入要生成的进程数量: ",
        lambda x: int(x) if int(x) > 0 else ValueError,
        "错误: 进程数量必须是正整数，请重新输入"
    )
    
    max_arrival_time = get_valid_input(
        "请输入最大到达时间: ",
        lambda x: int(x) if int(x) >= 0 else ValueError,
        "错误: 最大到达时间必须是非负整数，请重新输入"
    )
    
    max_run_time = get_valid_input(
        "请输入最大运行时间: ",
        lambda x: int(x) if int(x) > 0 else ValueError,
        "错误: 最大运行时间必须是正整数，请重新输入"
    )
    
    processes = []
    for i in range(num_processes):
        pid = f"P{i+1}"
        arrival_time = random.randint(0, max_arrival_time)
        run_time = random.randint(1, max_run_time)
        priority = random.randint(1, 5)  # 随机优先级1-5
        processes.append(Process(pid, arrival_time, run_time, priority))
    
    return processes

def select_scheduler(processes: List[Process]) -> Tuple[Scheduler, str]:
    """选择调度算法并配置参数"""
    print("\n=== 选择调度算法 ===")
    print("1. FCFS (先来先服务)")
    print("2. 非抢占式SJF (短作业优先)")
    print("3. 抢占式SJF (最短剩余时间优先, SRTF)")
    print("4. 非抢占式优先级调度")
    print("5. 抢占式优先级调度")
    print("6. 时间片轮转 (Round Robin, RR)")
    print("7. 多级队列调度 (Multilevel Queue Scheduling)")
    print("8. 多级反馈队列调度 (Multilevel Feedback Queue Scheduling)")
    
    choice = get_valid_input(
        "请选择调度算法 (1-8): ",
        lambda x: int(x) if 1 <= int(x) <= 8 else ValueError,
        "错误: 请输入1-8之间的数字"
    )
    
    scheduler = None
    algorithm_name = ""
    
    if choice == 1:
        scheduler = FCFS(processes)
        algorithm_name = "FCFS"
    
    elif choice == 2:
        scheduler = NonPreemptiveSJF(processes)
        algorithm_name = "非抢占式SJF"
    
    elif choice == 3:
        scheduler = PreemptiveSJF(processes)
        algorithm_name = "抢占式SJF (SRTF)"
    
    elif choice == 4:
        # 检查所有进程是否有优先级
        if any(p.priority is None for p in processes):
            print("错误: 非抢占式优先级调度要求所有进程都有优先级")
            # 为没有优先级的进程分配优先级
            for p in processes:
                if p.priority is None:
                    p.priority = get_valid_input(
                        f"请为进程 {p.pid} 输入优先级: ",
                        lambda x: int(x) if int(x) > 0 else ValueError,
                        "错误: 优先级必须是正整数"
                    )
        scheduler = NonPreemptivePriority(processes)
        algorithm_name = "非抢占式优先级调度"
    
    elif choice == 5:
        # 检查所有进程是否有优先级
        if any(p.priority is None for p in processes):
            print("错误: 抢占式优先级调度要求所有进程都有优先级")
            # 为没有优先级的进程分配优先级
            for p in processes:
                if p.priority is None:
                    p.priority = get_valid_input(
                        f"请为进程 {p.pid} 输入优先级: ",
                        lambda x: int(x) if int(x) > 0 else ValueError,
                        "错误: 优先级必须是正整数"
                    )
        scheduler = PreemptivePriority(processes)
        algorithm_name = "抢占式优先级调度"
    
    elif choice == 6:
        time_quantum = get_valid_input(
            "请输入时间片大小: ",
            lambda x: int(x) if int(x) > 0 else ValueError,
            "错误: 时间片大小必须是正整数"
        )
        scheduler = RoundRobin(processes, time_quantum)
        algorithm_name = f"时间片轮转 (RR, 时间片={time_quantum})"
    
    elif choice == 7:
        num_queues = get_valid_input(
            "请输入队列数量: ",
            lambda x: int(x) if 2 <= int(x) <= 5 else ValueError,
            "错误: 队列数量必须在2-5之间"
        )
        
        queue_config = []
        for i in range(num_queues):
            print(f"\n配置队列 {i+1}:")
            algorithm = get_valid_input(
                "请选择调度算法 (1=FCFS, 2=RR): ",
                lambda x: 'FCFS' if x == '1' else 'RR' if x == '2' else ValueError,
                "错误: 请输入1或2"
            )
            
            time_quantum = None
            if algorithm == 'RR':
                time_quantum = get_valid_input(
                    "请输入时间片大小: ",
                    lambda x: int(x) if int(x) > 0 else ValueError,
                    "错误: 时间片大小必须是正整数"
                )
            
            queue_config.append({
                'algorithm': algorithm,
                'time_quantum': time_quantum
            })
        
        scheduler = MultilevelQueue(processes, queue_config)
        algorithm_name = "多级队列调度"
    
    elif choice == 8:
        num_queues = get_valid_input(
            "请输入队列数量: ",
            lambda x: int(x) if 2 <= int(x) <= 5 else ValueError,
            "错误: 队列数量必须在2-5之间"
        )
        
        queue_config = []
        for i in range(num_queues):
            time_quantum = get_valid_input(
                f"请输入队列 {i+1} 的时间片大小 (最后一个队列将使用FCFS，时间片忽略): ",
                lambda x: int(x) if int(x) > 0 else ValueError,
                "错误: 时间片大小必须是正整数"
            )
            queue_config.append({'time_quantum': time_quantum})
        
        use_aging = get_valid_input(
            "是否启用老化机制? (1=是, 2=否): ",
            lambda x: True if x == '1' else False if x == '2' else ValueError,
            "错误: 请输入1或2"
        )
        
        aging_threshold = None
        if use_aging:
            aging_threshold = get_valid_input(
                "请输入老化阈值 (等待时间超过此值的进程将提升优先级): ",
                lambda x: int(x) if int(x) > 0 else ValueError,
                "错误: 老化阈值必须是正整数"
            )
        
        scheduler = MultilevelFeedbackQueue(processes, queue_config, aging_threshold)
        algorithm_name = "多级反馈队列调度"
        if use_aging:
            algorithm_name += f" (老化阈值={aging_threshold})"
    
    return scheduler, algorithm_name

def print_scheduler_info(scheduler: Scheduler, algorithm_name: str) -> None:
    """打印调度器信息和执行过程"""
    print(f"\n=== {algorithm_name} 调度 ===")
    
    # 打印进程信息
    print("\n进程信息:")
    print("PID\t到达时间\t运行时间\t优先级")
    for p in scheduler.processes:
        print(f"{p.pid}\t{p.arrival_time}\t\t{p.run_time}\t\t{p.priority if p.priority is not None else '-'}")
    
    # 运行调度算法
    scheduler.run()
    
    # 打印执行过程
    print("\n执行过程:")
    for history in scheduler.execution_history:
        if history['action'] == '空闲':
            print(f"时间 [{history['start_time']}-{history['end_time']}]: CPU 空闲")
        else:
            print(f"时间 [{history['start_time']}-{history['end_time']}]: 进程 [{history['pid']}] ({history['action']})")
    
    # 打印性能指标
    system_metrics, process_metrics = scheduler.get_metrics()
    
    print("\n=== 进程性能指标 ===")
    print("PID\t到达时间\t运行时间\t完成时间\t周转时间\t等待时间\t响应时间")
    for metrics in process_metrics:
        print(f"{metrics['进程ID']}\t{metrics['到达时间']}\t\t{metrics['运行时间']}\t\t"
              f"{metrics['完成时间']}\t\t{metrics['周转时间']}\t\t"
              f"{metrics['等待时间']}\t\t{metrics['响应时间']}")
    
    print("\n=== 系统整体性能指标 ===")
    print(f"CPU利用率: {system_metrics['CPU利用率']:.2%}")
    print(f"吞吐量: {system_metrics['吞吐量']:.2f} 进程/时间单位")
    print(f"平均周转时间: {system_metrics['平均周转时间']:.2f}")
    print(f"平均等待时间: {system_metrics['平均等待时间']:.2f}")
    print(f"平均响应时间: {system_metrics['平均响应时间']:.2f}")

def main() -> None:
    """主函数，程序入口点"""
    print("欢迎使用操作系统调度算法可视化工具！")
    
    # 选择进程数据输入方式
    print("\n=== 选择进程数据输入方式 ===")
    print("1. 手动输入")
    print("2. 随机生成")
    
    input_choice = get_valid_input(
        "请选择输入方式 (1-2): ",
        lambda x: int(x) if 1 <= int(x) <= 2 else ValueError,
        "错误: 请输入1或2"
    )
    
    # 获取进程数据
    if input_choice == 1:
        processes = manual_input_processes()
    else:
        processes = random_generate_processes()
    
    if not processes:
        print("错误: 没有输入任何进程")
        return
    
    # 选择调度算法
    scheduler, algorithm_name = select_scheduler(processes)
    
    # 打印调度信息和执行过程
    print_scheduler_info(scheduler, algorithm_name)
    
    # 可视化结果
    try:
        Visualizer.visualize(scheduler, algorithm_name)
    except Exception as e:
        print(f"\n警告: 可视化失败 ({str(e)})")
        print("请确保已安装plotly库 (pip install plotly)")

if __name__ == "__main__":
    main()        