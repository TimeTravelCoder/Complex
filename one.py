import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple, Union, Callable
import sys

class Process:
    """进程类，包含进程的各种属性和状态"""
    
    def __init__(self, pid: str, arrival_time: int, run_time: int, priority: Optional[int] = None):
        """初始化进程属性"""
        self.pid = pid  # 进程ID
        self.arrival_time = arrival_time  # 到达时间
        self.run_time = run_time  # 运行时间
        self.priority = priority  # 优先级，数值越小优先级越高
        
        # 动态属性
        self.remaining_time = run_time  # 剩余运行时间
        self.start_time = -1  # 首次执行时间
        self.completion_time = -1  # 完成时间
        self.waiting_time = 0  # 等待时间
        self.turnaround_time = 0  # 周转时间
        self.response_time = -1  # 响应时间
        
    def __str__(self):
        """进程信息的字符串表示"""
        return (f"进程ID: {self.pid}, 到达时间: {self.arrival_time}, "
                f"运行时间: {self.run_time}, 优先级: {self.priority}")

class Scheduler:
    """调度器基类，实现通用调度逻辑"""
    
    def __init__(self, processes: List[Process]):
        """初始化调度器"""
        self.processes = processes.copy()  # 进程列表
        self.ready_queue: List[Process] = []  # 就绪队列
        self.current_time = 0  # 当前时间
        self.running_process: Optional[Process] = None  # 当前运行的进程
        self.execution_history: List[Dict[str, Union[int, str]]] = []  # 执行历史
        self.completed_processes: List[Process] = []  # 已完成的进程
        self.total_idle_time = 0  # CPU空闲时间
        self.last_idle_start = 0  # 上次空闲开始时间
        
    def update_ready_queue(self):
        """更新就绪队列，将已到达且未完成的进程加入就绪队列"""
        # 检查是否有新进程到达
        for process in self.processes:
            if (process.arrival_time <= self.current_time and 
                process not in self.ready_queue and 
                process not in self.completed_processes and
                process != self.running_process):
                self.ready_queue.append(process)
    
    def select_process(self) -> Optional[Process]:
        """选择要执行的进程，由子类实现"""
        raise NotImplementedError("子类必须实现select_process方法")
    
    def execute_process(self, process: Process, time_quantum: Optional[int] = None):
        """执行进程一个时间单位或指定的时间片"""
        # 记录首次执行时间
        if process.start_time == -1:
            process.start_time = self.current_time
            process.response_time = self.current_time - process.arrival_time
        
        # 执行一个时间单位
        execution_time = 1
        if time_quantum is not None:
            execution_time = min(time_quantum, process.remaining_time)
        
        # 更新剩余时间
        process.remaining_time -= execution_time
        
        # 记录执行历史
        start_time = self.current_time
        self.current_time += execution_time
        
        # 检查进程是否完成
        if process.remaining_time == 0:
            process.completion_time = self.current_time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.run_time
            self.completed_processes.append(process)
            self.execution_history.append({
                'start_time': start_time,
                'end_time': self.current_time,
                'pid': process.pid,
                'status': '完成'
            })
            self.running_process = None
        else:
            self.execution_history.append({
                'start_time': start_time,
                'end_time': self.current_time,
                'pid': process.pid,
                'status': '执行'
            })
            self.running_process = process
    
    def run(self) -> Dict[str, Union[float, int]]:
        """运行调度算法，返回性能指标"""
        # 主调度循环
        while len(self.completed_processes) < len(self.processes):
            self.update_ready_queue()
            
            # 检查是否有进程在运行
            if self.running_process is None or self.running_process.remaining_time == 0:
                # 如果就绪队列为空，CPU空闲
                if not self.ready_queue:
                    self.total_idle_time += 1
                    self.current_time += 1
                    continue
                
                # 选择下一个要执行的进程
                self.running_process = self.select_process()
                if self.running_process and self.running_process in self.ready_queue:
                    self.ready_queue.remove(self.running_process)
                    self.execute_process(self.running_process)
            else:
                # 继续执行当前进程
                self.execute_process(self.running_process)
        
        # 计算性能指标
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict[str, Union[float, int]]:
        """计算性能指标"""
        total_turnaround_time = sum(p.turnaround_time for p in self.completed_processes)
        total_waiting_time = sum(p.waiting_time for p in self.completed_processes)
        total_response_time = sum(p.response_time for p in self.completed_processes)
        
        cpu_utilization = (self.current_time - self.total_idle_time) / self.current_time * 100
        throughput = len(self.completed_processes) / self.current_time
        
        return {
            'cpu_utilization': cpu_utilization,
            'throughput': throughput,
            'avg_turnaround_time': total_turnaround_time / len(self.completed_processes),
            'avg_waiting_time': total_waiting_time / len(self.completed_processes),
            'avg_response_time': total_response_time / len(self.completed_processes),
            'total_time': self.current_time
        }
    
    def print_execution_history(self):
        """打印执行历史"""
        print("\n执行序列:")
        for entry in self.execution_history:
            print(f"时间 [{entry['start_time']}-{entry['end_time']}]: 进程 [{entry['pid']}] ({entry['status']})")
    
    def print_process_details(self):
        """打印进程详细信息"""
        print("\n进程详细信息:")
        print(f"{'进程ID':<8}{'到达时间':<10}{'运行时间':<10}{'完成时间':<10}"
              f"{'周转时间':<10}{'等待时间':<10}{'响应时间':<10}")
        for process in sorted(self.completed_processes, key=lambda p: p.arrival_time):
            print(f"{process.pid:<8}{process.arrival_time:<10}{process.run_time:<10}"
                  f"{process.completion_time:<10}{process.turnaround_time:<10}"
                  f"{process.waiting_time:<10}{process.response_time:<10}")
    
    def visualize(self, algorithm_name: str, params: Optional[Dict[str, str]] = None):
        """使用Plotly可视化调度结果"""
        # 创建甘特图数据
        colors = {}
        pids = sorted(list({entry['pid'] for entry in self.execution_history}))
        for i, pid in enumerate(pids):
            colors[pid] = f"hsl({i * 360 / len(pids)}, 70%, 70%)"
        
        tasks = []
        for entry in self.execution_history:
            tasks.append({
                'Task': entry['pid'],
                'Start': entry['start_time'],
                'Finish': entry['end_time'],
                'Resource': entry['pid']
            })
        
        # 创建甘特图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"{algorithm_name} 调度甘特图", "性能指标"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # 添加甘特图
        for pid in pids:
            pid_tasks = [t for t in tasks if t['Resource'] == pid]
            for task in pid_tasks:
                fig.add_trace(
                    go.Bar(
                        x=[task['Finish'] - task['Start']],
                        y=[task['Task']],
                        orientation='h',
                        marker=dict(color=colors[pid]),
                        base=task['Start'],
                        showlegend=False,
                        hovertemplate=f"进程: {task['Task']}<br>开始时间: {task['Start']}<br>结束时间: {task['Finish']}<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        # 设置甘特图布局
        fig.update_layout(
            title={
                'text': f"{algorithm_name} 调度结果可视化",
                'font': {'family': 'SimHei', 'size': 20},
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='时间',
                title_font={'family': 'SimHei', 'size': 16},
                tickfont={'family': 'SimHei', 'size': 14},
                dtick=1,
                showgrid=True
            ),
            yaxis=dict(
                title='进程',
                title_font={'family': 'SimHei', 'size': 16},
                tickfont={'family': 'SimHei', 'size': 14},
                categoryorder='array',
                categoryarray=pids,
                showgrid=True
            ),
            height=800,
            width=1000,
            margin=dict(l=100, r=100, t=100, b=100)
        )
        
        # 添加性能指标表格
        metrics = self.calculate_metrics()
        
        # 构建参数文本
        param_text = ""
        if params:
            param_text = "<br>".join([f"{k}: {v}" for k, v in params.items()])
            param_text = f"<b>参数:</b><br>{param_text}<br><br>"
        
        # 构建指标文本
        metrics_text = (f"<b>CPU利用率:</b> {metrics['cpu_utilization']:.2f}%<br>"
                        f"<b>吞吐量:</b> {metrics['throughput']:.2f} 进程/时间单位<br>"
                        f"<b>平均周转时间:</b> {metrics['avg_turnaround_time']:.2f}<br>"
                        f"<b>平均等待时间:</b> {metrics['avg_waiting_time']:.2f}<br>"
                        f"<b>平均响应时间:</b> {metrics['avg_response_time']:.2f}<br>"
                        f"<b>总时间:</b> {metrics['total_time']}")
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["指标", "值"],
                    font=dict(family="SimHei", size=14),
                    align="center"
                ),
                cells=dict(
                    values=[
                        ["CPU利用率", "吞吐量", "平均周转时间", "平均等待时间", "平均响应时间", "总时间"],
                        [f"{metrics['cpu_utilization']:.2f}%", 
                         f"{metrics['throughput']:.2f} 进程/时间单位",
                         f"{metrics['avg_turnaround_time']:.2f}",
                         f"{metrics['avg_waiting_time']:.2f}",
                         f"{metrics['avg_response_time']:.2f}",
                         f"{metrics['total_time']}"]
                    ],
                    font=dict(family="SimHei", size=14),
                    align="center"
                )
            ),
            row=2, col=1
        )
        
        # 更新第二行布局
        fig.update_layout(
            font=dict(family="SimHei"),
            annotations=[
                dict(
                    text=param_text,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(family="SimHei", size=14)
                )
            ]
        )
        
        # 显示图表
        fig.show()

# 各种调度算法的实现
class FCFS(Scheduler):
    """先来先服务(FCFS)调度算法"""
    
    def select_process(self) -> Optional[Process]:
        """选择到达时间最早的进程"""
        if not self.ready_queue:
            return None
        # 按到达时间排序，到达时间相同则按进程ID排序
        return sorted(self.ready_queue, key=lambda p: (p.arrival_time, p.pid))[0]

class NonPreemptiveSJF(Scheduler):
    """非抢占式短作业优先(SJF)调度算法"""
    
    def select_process(self) -> Optional[Process]:
        """选择运行时间最短的进程"""
        if not self.ready_queue:
            return None
        # 按运行时间排序，运行时间相同则按到达时间排序
        return sorted(self.ready_queue, key=lambda p: (p.run_time, p.arrival_time))[0]

class PreemptiveSJF(Scheduler):
    """抢占式短作业优先(SRTF)调度算法"""
    
    def update_ready_queue(self):
        """更新就绪队列，检查是否需要抢占当前进程"""
        super().update_ready_queue()
        
        # 检查是否需要抢占当前进程
        if (self.running_process and 
            self.ready_queue and 
            any(p.remaining_time < self.running_process.remaining_time for p in self.ready_queue)):
            # 将当前进程放回就绪队列（如果不在队列中）
            if self.running_process not in self.ready_queue:
                self.ready_queue.append(self.running_process)
            self.execution_history.append({
                'start_time': self.current_time,
                'end_time': self.current_time,
                'pid': self.running_process.pid,
                'status': '抢占'
            })
            self.running_process = None
    
    def select_process(self) -> Optional[Process]:
        """选择剩余运行时间最短的进程"""
        if not self.ready_queue:
            return None
        # 按剩余时间排序
        return sorted(self.ready_queue, key=lambda p: p.remaining_time)[0]

class NonPreemptivePriority(Scheduler):
    """非抢占式优先级调度算法"""
    
    def select_process(self) -> Optional[Process]:
        """选择优先级最高的进程"""
        if not self.ready_queue:
            return None
        # 按优先级排序，优先级相同则按到达时间排序
        return sorted(self.ready_queue, key=lambda p: (p.priority, p.arrival_time))[0]

class PreemptivePriority(Scheduler):
    """抢占式优先级调度算法"""
    
    def update_ready_queue(self):
        """更新就绪队列，检查是否需要抢占当前进程"""
        super().update_ready_queue()
        
        # 检查是否需要抢占当前进程
        if (self.running_process and 
            self.ready_queue and 
            any(p.priority < self.running_process.priority for p in self.ready_queue)):
            # 将当前进程放回就绪队列（如果不在队列中）
            if self.running_process not in self.ready_queue:
                self.ready_queue.append(self.running_process)
            self.execution_history.append({
                'start_time': self.current_time,
                'end_time': self.current_time,
                'pid': self.running_process.pid,
                'status': '抢占'
            })
            self.running_process = None
    
    def select_process(self) -> Optional[Process]:
        """选择优先级最高的进程"""
        if not self.ready_queue:
            return None
        # 按优先级排序
        return sorted(self.ready_queue, key=lambda p: p.priority)[0]

class RoundRobin(Scheduler):
    """时间片轮转(RR)调度算法"""
    
    def __init__(self, processes: List[Process], time_quantum: int):
        """初始化RR调度器"""
        super().__init__(processes)
        self.time_quantum = time_quantum
    
    def select_process(self) -> Optional[Process]:
        """从就绪队列头部选择进程"""
        if not self.ready_queue:
            return None
        return self.ready_queue.pop(0)
    
    def execute_process(self, process: Process, time_quantum: Optional[int] = None):
        """执行进程一个时间片"""
        # 如果未指定时间片，使用RR的时间片
        if time_quantum is None:
            time_quantum = self.time_quantum
        
        super().execute_process(process, time_quantum)
        
        # 如果进程未完成，将其放回就绪队列尾部
        if process in self.processes and process not in self.completed_processes:
            self.ready_queue.append(process)

class MultilevelQueue(Scheduler):
    """多级队列调度算法"""
    
    def __init__(self, processes: List[Process], queues_info: List[Dict[str, Union[str, int]]]):
        """初始化多级队列调度器"""
        super().__init__(processes)
        self.queues: List[List[Process]] = [[] for _ in range(len(queues_info))]
        self.queues_info = queues_info
        self.current_queue_index = 0
        
        # 为每个进程分配队列
        for process in processes:
            queue_index = int(process.pid[1:]) % len(queues_info)  # 使用进程ID的数字部分取模
            self.queues[queue_index].append(process)
    
    def update_ready_queue(self):
        """更新就绪队列，从高优先级队列开始处理"""
        # 清空当前就绪队列
        self.ready_queue = []
        
        # 从高优先级队列开始，将所有队列中的就绪进程加入就绪队列
        for i, queue in enumerate(self.queues):
            for process in queue:
                if (process.arrival_time <= self.current_time and 
                    process not in self.completed_processes and
                    process != self.running_process):
                    self.ready_queue.append(process)
            
            # 如果当前队列有就绪进程，不再处理低优先级队列
            if self.ready_queue:
                self.current_queue_index = i
                break
    
    def select_process(self) -> Optional[Process]:
        """根据当前队列的调度算法选择进程"""
        if not self.ready_queue:
            return None
        
        algorithm = self.queues_info[self.current_queue_index]['algorithm']
        
        if algorithm == 'FCFS':
            # 按到达时间排序
            return sorted(self.ready_queue, key=lambda p: (p.arrival_time, p.pid))[0]
        elif algorithm == 'RR':
            # RR算法，从队列头部取出
            time_quantum = self.queues_info[self.current_queue_index]['time_quantum']
            process = self.ready_queue.pop(0)
            # 如果进程未完成，放回队列尾部
            if process.remaining_time > time_quantum:
                self.ready_queue.append(process)
            return process
        else:
            # 默认使用FCFS
            return sorted(self.ready_queue, key=lambda p: (p.arrival_time, p.pid))[0]
    
    def execute_process(self, process: Process, time_quantum: Optional[int] = None):
        """执行进程"""
        # 如果是RR队列，使用指定的时间片
        if self.queues_info[self.current_queue_index]['algorithm'] == 'RR':
            time_quantum = self.queues_info[self.current_queue_index]['time_quantum']
        
        super().execute_process(process, time_quantum)

class MultilevelFeedbackQueue(Scheduler):
    """多级反馈队列调度算法"""
    
    def __init__(self, processes: List[Process], queues_info: List[Dict[str, Union[str, int]]], 
                 aging_threshold: Optional[int] = None):
        """初始化多级反馈队列调度器"""
        super().__init__(processes)
        self.queues: List[List[Process]] = [[] for _ in range(len(queues_info))]
        self.queues_info = queues_info
        self.current_queue_index = 0
        self.aging_threshold = aging_threshold
        self.waiting_time_in_queue: Dict[Process, int] = {}  # 记录进程在当前队列的等待时间
        
        # 所有进程初始都在最高优先级队列
        for process in processes:
            self.queues[0].append(process)
            self.waiting_time_in_queue[process] = 0
    
    def update_ready_queue(self):
        """更新就绪队列，处理老化和队列间移动"""
        # 清空当前就绪队列
        self.ready_queue = []
        
        # 应用老化机制
        if self.aging_threshold is not None:
            for i in range(len(self.queues) - 1, 0, -1):  # 从低优先级队列到高优先级队列
                for process in self.queues[i]:
                    if (process.arrival_time <= self.current_time and 
                        process not in self.completed_processes and
                        process != self.running_process):
                        self.waiting_time_in_queue[process] += 1
                        if self.waiting_time_in_queue[process] >= self.aging_threshold:
                            # 提升到更高优先级队列
                            self.queues[i].remove(process)
                            self.queues[i-1].append(process)
                            self.waiting_time_in_queue[process] = 0
        
        # 从高优先级队列开始，将所有队列中的就绪进程加入就绪队列
        for i, queue in enumerate(self.queues):
            for process in queue:
                if (process.arrival_time <= self.current_time and 
                    process not in self.completed_processes and
                    process != self.running_process):
                    self.ready_queue.append(process)
            
            # 如果当前队列有就绪进程，不再处理低优先级队列
            if self.ready_queue:
                self.current_queue_index = i
                break
    
    def select_process(self) -> Optional[Process]:
        """根据当前队列的调度算法选择进程"""
        if not self.ready_queue:
            return None
        
        # 从当前队列头部选择进程
        return self.ready_queue.pop(0)
    
    def execute_process(self, process: Process, time_quantum: Optional[int] = None):
        """执行进程，处理时间片和队列间移动"""
        # 使用当前队列的时间片
        time_quantum = self.queues_info[self.current_queue_index]['time_quantum']
        
        # 记录进程开始执行前所在的队列
        original_queue_index = self.current_queue_index
        
        super().execute_process(process, time_quantum)
        
        # 如果进程未完成，将其移至下一级队列
        if (process in self.processes and 
            process not in self.completed_processes and
            original_queue_index < len(self.queues) - 1):
            # 从当前队列移除
            if process in self.queues[original_queue_index]:
                self.queues[original_queue_index].remove(process)
            # 添加到下一级队列
            self.queues[original_queue_index + 1].append(process)
            self.waiting_time_in_queue[process] = 0

# 用户交互界面
class SchedulerUI:
    """调度器用户界面"""
    
    def __init__(self):
        """初始化用户界面"""
        self.processes: List[Process] = []
    
    def get_valid_input(self, prompt: str, valid_type: type, valid_range: Optional[Tuple] = None) -> any:
        """获取用户有效输入"""
        while True:
            try:
                value = valid_type(input(prompt))
                if valid_range is not None:
                    min_val, max_val = valid_range
                    if not (min_val <= value <= max_val):
                        print(f"输入值必须在 {min_val} 到 {max_val} 之间!")
                        continue
                return value
            except ValueError:
                print(f"无效输入! 请输入{valid_type.__name__}类型的值.")
    
    def input_processes_manually(self) -> List[Process]:
        """手动输入进程信息"""
        processes = []
        print("\n手动输入进程信息 (输入空的进程ID结束)")
        
        while True:
            pid = input("请输入进程ID (如P1): ").strip()
            if not pid:
                break
            
            arrival_time = self.get_valid_input("请输入到达时间: ", int, (0, float('inf')))
            run_time = self.get_valid_input("请输入运行时间: ", int, (1, float('inf')))
            
            # 询问是否需要输入优先级
            need_priority = input("是否需要输入优先级? (y/n): ").strip().lower() == 'y'
            priority = None
            if need_priority:
                priority = self.get_valid_input("请输入优先级: ", int, (1, float('inf')))
            
            processes.append(Process(pid, arrival_time, run_time, priority))
            print(f"进程 {pid} 已添加")
        
        return processes
    
    def generate_random_processes(self) -> List[Process]:
        """随机生成进程信息"""
        print("\n随机生成进程信息")
        num_processes = self.get_valid_input("请输入要生成的进程数量: ", int, (1, 20))
        max_arrival_time = self.get_valid_input("请输入最大到达时间: ", int, (1, 100))
        max_run_time = self.get_valid_input("请输入最大运行时间: ", int, (1, 20))
        
        processes = []
        for i in range(num_processes):
            pid = f"P{i+1}"
            arrival_time = random.randint(0, max_arrival_time)
            run_time = random.randint(1, max_run_time)
            priority = random.randint(1, 10)  # 随机优先级1-10
            processes.append(Process(pid, arrival_time, run_time, priority))
        
        return processes
    
    def select_algorithm(self) -> Tuple[Callable, Dict[str, str]]:
        """选择调度算法"""
        print("\n请选择调度算法:")
        print("1. FCFS (先来先服务)")
        print("2. 非抢占式SJF (短作业优先)")
        print("3. 抢占式SJF (最短剩余时间优先, SRTF)")
        print("4. 非抢占式优先级调度")
        print("5. 抢占式优先级调度")
        print("6. 时间片轮转 (Round Robin, RR)")
        print("7. 多级队列调度")
        print("8. 多级反馈队列调度")
        
        choice = self.get_valid_input("请输入选择 (1-8): ", int, (1, 8))
        params = {}
        
        if choice == 6:  # RR
            time_quantum = self.get_valid_input("请输入时间片大小: ", int, (1, float('inf')))
            params["时间片"] = str(time_quantum)
            return lambda processes: RoundRobin(processes, time_quantum), params
        
        elif choice == 7:  # 多级队列调度
            num_queues = self.get_valid_input("请输入队列数量: ", int, (2, 5))
            queues_info = []
            
            for i in range(num_queues):
                print(f"\n配置队列 {i+1}:")
                print("1. FCFS")
                print("2. RR")
                algo_choice = self.get_valid_input(f"请选择队列 {i+1} 的调度算法 (1-2): ", int, (1, 2))
                
                algo = 'FCFS' if algo_choice == 1 else 'RR'
                queue_params = {'algorithm': algo}
                
                if algo == 'RR':
                    time_quantum = self.get_valid_input(f"请输入队列 {i+1} 的时间片大小: ", int, (1, float('inf')))
                    queue_params['time_quantum'] = time_quantum
                
                queues_info.append(queue_params)
            
            params["队列配置"] = str(queues_info)
            return lambda processes: MultilevelQueue(processes, queues_info), params
        
        elif choice == 8:  # 多级反馈队列调度
            num_queues = self.get_valid_input("请输入队列数量: ", int, (2, 5))
            queues_info = []
            
            for i in range(num_queues):
                time_quantum = self.get_valid_input(f"请输入队列 {i+1} 的时间片大小 (最后一级队列时间片会被忽略): ", 
                                                  int, (1, float('inf')))
                queues_info.append({'time_quantum': time_quantum})
            
            use_aging = input("是否启用老化机制? (y/n): ").strip().lower() == 'y'
            aging_threshold = None
            if use_aging:
                aging_threshold = self.get_valid_input("请输入老化阈值 (时间单位): ", int, (1, float('inf')))
            
            params["队列配置"] = str([qi['time_quantum'] for qi in queues_info])
            params["是否老化"] = str(use_aging)
            if use_aging:
                params["老化阈值"] = str(aging_threshold)
            
            return lambda processes: MultilevelFeedbackQueue(processes, queues_info, aging_threshold), params
        
        else:
            # 不需要额外参数的算法
            algorithms = {
                1: lambda processes: FCFS(processes),
                2: lambda processes: NonPreemptiveSJF(processes),
                3: lambda processes: PreemptiveSJF(processes),
                4: lambda processes: NonPreemptivePriority(processes),
                5: lambda processes: PreemptivePriority(processes)
            }
            
            algo_names = {
                1: "FCFS",
                2: "非抢占式SJF",
                3: "抢占式SJF",
                4: "非抢占式优先级",
                5: "抢占式优先级"
            }
            
            return algorithms[choice], {"算法": algo_names[choice]}
    
    def check_all_processes_have_priority(self) -> bool:
        """检查所有进程是否都有优先级"""
        return all(p.priority is not None for p in self.processes)
    
    def add_priority_to_processes(self):
        """为所有进程添加优先级"""
        print("\n所有进程都需要有优先级才能运行此调度算法")
        for process in self.processes:
            if process.priority is None:
                priority = self.get_valid_input(f"请为进程 {process.pid} 设置优先级: ", int, (1, float('inf')))
                process.priority = priority
    
    def run(self):
        """运行调度器UI"""
        print("欢迎使用操作系统调度算法可视化工具!")
        
        # 输入进程数据
        print("\n请选择进程数据输入方式:")
        print("1. 手动输入")
        print("2. 随机生成")
        input_choice = self.get_valid_input("请输入选择 (1-2): ", int, (1, 2))
        
        if input_choice == 1:
            self.processes = self.input_processes_manually()
        else:
            self.processes = self.generate_random_processes()
        
        if not self.processes:
            print("没有进程数据，程序退出!")
            return
        
        # 选择调度算法
        scheduler_factory, params = self.select_algorithm()
        
        # 检查是否需要优先级
        if (params.get("算法", "").endswith("优先级") or 
            isinstance(scheduler_factory(self.processes), (MultilevelQueue, MultilevelFeedbackQueue))):
            if not self.check_all_processes_have_priority():
                self.add_priority_to_processes()
        
        # 显示进程信息
        print("\n进程信息:")
        for process in self.processes:
            print(process)
        
        # 创建并运行调度器
        scheduler = scheduler_factory(self.processes)
        
        # 获取算法名称
        algorithm_name = type(scheduler).__name__
        if algorithm_name == 'RoundRobin':
            algorithm_name = f"时间片轮转 (RR, 时间片={params['时间片']})"
        elif algorithm_name == 'MultilevelQueue':
            algorithm_name = "多级队列调度"
        elif algorithm_name == 'MultilevelFeedbackQueue':
            algorithm_name = "多级反馈队列调度"
        
        # 运行调度
        metrics = scheduler.run()
        
        # 输出结果
        scheduler.print_execution_history()
        scheduler.print_process_details()
        
        print("\n系统整体性能指标:")
        print(f"CPU利用率: {metrics['cpu_utilization']:.2f}%")
        print(f"吞吐量: {metrics['throughput']:.2f} 进程/时间单位")
        print(f"平均周转时间: {metrics['avg_turnaround_time']:.2f}")
        print(f"平均等待时间: {metrics['avg_waiting_time']:.2f}")
        print(f"平均响应时间: {metrics['avg_response_time']:.2f}")
        
        # 可视化
        scheduler.visualize(algorithm_name, params)

if __name__ == "__main__":
    try:
        ui = SchedulerUI()
        ui.run()
    except KeyboardInterrupt:
        print("\n程序已退出!")
        sys.exit(0)    