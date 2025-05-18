import heapq
import random
import math
from typing import List, Dict, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

# --- Add warning filter ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly.express._core")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.groupby.generic")

# --- Process Class ---
class Process:
    def __init__(self, pid: str, arrival_time: int, run_time: int, priority: Optional[int] = None):
        if run_time <= 0:
            raise ValueError("运行时间必须为正整数")
        self.pid: str = pid
        self.arrival_time: int = arrival_time
        self.run_time: int = run_time
        # Lower numerical value means higher priority for scheduler logic
        self.priority: Optional[int] = priority

        self.remaining_time: int = run_time
        self.start_time: Optional[int] = None
        self.completion_time: Optional[int] = None
        self.waiting_time: int = 0
        self.turnaround_time: int = 0
        self.response_time: Optional[int] = None

        # For MLFQ
        self.current_queue_level: int = 0
        self.time_in_current_queue_slice: int = 0
        self.last_run_time: int = 0
        self.assigned_queue_index: int = 0 # For MLQ

    def __lt__(self, other: 'Process') -> bool:
        return self.arrival_time < other.arrival_time

    def __repr__(self) -> str:
        priority_display = self.priority if self.priority is not None else 'N/A'
        return (f"Process(pid={self.pid}, arrival={self.arrival_time}, run={self.run_time}, "
                f"priority={priority_display}, remaining={self.remaining_time})")

    def get_priority_val(self) -> float:
        """Returns the priority value, defaulting to infinity for undefined priorities."""
        return self.priority if self.priority is not None else float('inf')

    def reset(self):
        self.remaining_time = self.run_time
        self.start_time = None
        self.completion_time = None
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = None
        self.current_queue_level = 0
        self.time_in_current_queue_slice = 0
        self.last_run_time = 0

# --- Scheduler Base Class ---
class Scheduler:
    def __init__(self, processes: List[Process]):
        self.processes_master_list: List[Process] = sorted([p for p in processes], key=lambda p: (p.arrival_time, p.pid))
        for p in self.processes_master_list:
            p.reset()

        self.completed_processes: List[Process] = []
        self.gantt_chart_data: List[Dict[str, Any]] = []
        self.current_time: int = 0
        self.total_cpu_busy_time: int = 0
        self.ready_queue: Any = []
        self.algorithm_name: str = "Scheduler Base"
        self.algorithm_params: Dict[str, Any] = {}

    def _update_processes_to_ready_state(self, pending_processes_list: List[Process]) -> List[Process]:
        newly_ready_processes = []
        for process in list(pending_processes_list):
            if process.arrival_time <= self.current_time:
                self._enqueue_process(process)
                pending_processes_list.remove(process)
                newly_ready_processes.append(process)
        return newly_ready_processes

    def _enqueue_process(self, process: Process) -> None:
        if process not in self.ready_queue:
             self.ready_queue.append(process)

    def _calculate_metrics(self, process: Process) -> None:
        if process.completion_time is not None and process.start_time is not None:
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.run_time
            if process.response_time is None:
                process.response_time = process.start_time - process.arrival_time
        else:
            print(f"警告: 进程 {process.pid} 的指标计算被跳过，因其开始或完成时间缺失。")

    def _calculate_system_metrics(self) -> Dict[str, float]:
        if not self.completed_processes:
            return {
                "CPU Utilization": 0.0, "Throughput": 0.0,
                "Average Turnaround Time": 0.0, "Average Waiting Time": 0.0,
                "Average Response Time": 0.0
            }

        total_turnaround_time = sum(p.turnaround_time for p in self.completed_processes)
        total_waiting_time = sum(p.waiting_time for p in self.completed_processes)
        total_response_time = sum(p.response_time for p in self.completed_processes if p.response_time is not None)

        num_completed = len(self.completed_processes)
        max_completion_time = max(p.completion_time for p in self.completed_processes if p.completion_time is not None)
        total_simulation_time = max(1, max_completion_time)

        cpu_utilization = (self.total_cpu_busy_time / total_simulation_time) * 100
        throughput = num_completed / total_simulation_time

        return {
            "CPU Utilization": round(cpu_utilization, 2),
            "Throughput": round(throughput, 2),
            "Average Turnaround Time": round(total_turnaround_time / num_completed, 2),
            "Average Waiting Time": round(total_waiting_time / num_completed, 2),
            "Average Response Time": round(total_response_time / num_completed, 2)
        }

    def run(self) -> None:
        raise NotImplementedError("The run method must be implemented by subclasses.")

    def _add_gantt_entry(self, process_pid: str, start: int, finish: int, status: str = "Executing"):
        print(f"时间 [{start}-{finish}]: 进程 [{process_pid}] ({status})")

        if finish <= start:
            if status == "Executing":
                print(f"  (Gantt: 执行事件 {process_pid} 时长为0或负, 不添加到甘特图)")
                return
            else:
                print(f"  (Gantt: 非执行事件 {process_pid} 时长为0或负, 简化为单点显示)")
                # 对于非执行事件，强制显示为一个时间点
                self.gantt_chart_data.append({
                    "Task": process_pid,
                    "Start": start,
                    "Finish": start + 0.1,  # 显示为一个小线段
                    "Resource": process_pid,
                    "Status": status
                })
                return

        self.gantt_chart_data.append({
            "Task": process_pid,
            "Start": start,
            "Finish": finish,
            "Resource": process_pid,
            "Status": status
        })

    def print_initial_processes(self):
        print("\n初始进程信息:")
        for p in self.processes_master_list:
            priority_display = p.priority if p.priority is not None else 'N/A'
            print(f"  PID: {p.pid}, 到达时间: {p.arrival_time}, 运行时间: {p.run_time}, "
                  f"优先级: {priority_display}")
        print("-" * 30)

    def print_results(self) -> None:
        print(f"\n--- {self.algorithm_name} 调度结果 ---")
        if self.algorithm_params:
            print("算法参数:")
            for key, value in self.algorithm_params.items():
                print(f"  {key}: {value}")

        print("\n各进程调度详情:")
        completed_map = {p.pid: p for p in self.completed_processes}

        for p_master in self.processes_master_list:
            p = completed_map.get(p_master.pid)
            if p and p.completion_time is not None:
                priority_display = p.priority if p.priority is not None else "N/A"
                response_time_display = p.response_time if p.response_time is not None else "N/A"
                start_time_display = p.start_time if p.start_time is not None else "N/A"
                print(f"  进程ID: {p.pid}, 到达: {p.arrival_time}, 运行: {p.run_time}, "
                      f"优先级: {priority_display}, 开始执行: {start_time_display}, 完成: {p.completion_time}, "
                      f"周转: {p.turnaround_time}, 等待: {p.waiting_time}, 响应: {response_time_display}")
            else:
                print(f"  进程ID: {p_master.pid} (到达: {p_master.arrival_time}, 运行: {p_master.run_time}) 未完成或数据不完整。")

        system_metrics = self._calculate_system_metrics()
        print("\n系统整体性能指标:")
        print(f"  CPU 利用率: {system_metrics['CPU Utilization']}%")
        print(f"  吞吐量: {system_metrics['Throughput']} 进程/单位时间")
        print(f"  平均周转时间: {system_metrics['Average Turnaround Time']}")
        print(f"  平均等待时间: {system_metrics['Average Waiting Time']}")
        print(f"  平均响应时间: {system_metrics['Average Response Time']}")

    def plot_gantt_and_metrics(self) -> None:
        gantt_possible_initially = bool(self.gantt_chart_data)
        df = pd.DataFrame()

        if gantt_possible_initially:
            print(f"\n原始甘特图数据点数量: {len(self.gantt_chart_data)}")
            df = pd.DataFrame(self.gantt_chart_data)
            df['Start'] = pd.to_numeric(df['Start'], errors='coerce')
            df['Finish'] = pd.to_numeric(df['Finish'], errors='coerce')
            df.dropna(subset=['Start', 'Finish'], inplace=True)

            # 改进的过滤逻辑：仅过滤执行事件中的无效时长，保留其他状态的瞬时事件
            mask = (df['Status'].isin(["Executing", "完成"])) & (df['Finish'] <= df['Start'])
            filtered_df = df[~mask]

            if len(filtered_df) < len(df):
                print(f"甘特图数据从 {len(df)} 条过滤为 {len(filtered_df)} 条 (移除了执行事件中的无效时长)")
            df = filtered_df

        gantt_display_possible = not df.empty

        if not gantt_display_possible:
            print("没有有效的甘特图数据可供显示。将仅显示性能指标表格。")
            if not gantt_possible_initially and not self.completed_processes:
                 print("并且似乎没有任何进程完成，可能模拟未正确执行或无进程输入。")
                 return

        # Create metrics table data
        system_metrics = self._calculate_system_metrics()
        metrics_names = ["CPU 利用率 (%)", "吞吐量 (个/单位时间)", "平均周转时间", "平均等待时间", "平均响应时间"]
        metrics_values = [
            system_metrics["CPU Utilization"], system_metrics["Throughput"],
            system_metrics["Average Turnaround Time"], system_metrics["Average Waiting Time"],
            system_metrics["Average Response Time"]
        ]

        # Determine number of rows for subplots
        rows = 1
        row_heights = [1.0]
        specs = [[{"type": "table"}]]
        subplot_titles_list = [f"{self.algorithm_name} 系统性能指标"]

        if gantt_display_possible:
            rows = 2
            row_heights = [0.7, 0.3]
            specs = [[{"type": "xy"}], [{"type": "table"}]]
            subplot_titles_list = (f"{self.algorithm_name} 调度甘特图", f"{self.algorithm_name} 系统性能指标")

        final_fig = make_subplots(
            rows=rows, cols=1,
            row_heights=row_heights,
            specs=specs,
            vertical_spacing=0.15 if rows > 1 else 0.05,
            subplot_titles=subplot_titles_list
        )

        gantt_chart_row_index = 1
        metrics_table_row_index = 1

        if gantt_display_possible:
            fig_gantt_internal = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource")
            for trace in fig_gantt_internal.data:
                final_fig.add_trace(trace, row=gantt_chart_row_index, col=1)

            # 确保时间轴起点不小于0
            min_start_time = max(df['Start'].min(), 0)
            max_finish_time = df['Finish'].max() if not df.empty else 0
            tickvals = list(range(math.floor(min_start_time), math.ceil(max_finish_time) + 2))

            final_fig.update_xaxes(
                title_text="时间",
                row=gantt_chart_row_index, col=1,
                type='linear',
                tickmode='array',
                tickvals=tickvals,
                range=[min_start_time - 0.5, max_finish_time + 0.5]
            )
            
            final_fig.update_yaxes(title_text="进程", row=gantt_chart_row_index, col=1, categoryorder="total ascending")
            
            # 添加状态标签
            for idx, row in df.iterrows():
                final_fig.add_annotation(
                    x=(row['Start'] + row['Finish']) / 2,
                    y=row['Task'],
                    text=row['Status'],
                    showarrow=False,
                    font=dict(size=10, color='black')
                )
                
            metrics_table_row_index = 2

        # Create metrics table trace
        table_trace = go.Table(
            header=dict(values=['性能指标', '数值'],
                        fill_color='paleturquoise',
                        align='left', font=dict(family="SimHei, Noto Sans CJK SC, Microsoft YaHei, sans-serif")),
            cells=dict(values=[metrics_names, metrics_values],
                       fill_color='lavender',
                       align='left', font=dict(family="SimHei, Noto Sans CJK SC, Microsoft YaHei, sans-serif"))
        )
        final_fig.add_trace(table_trace, row=metrics_table_row_index, col=1)

        final_fig.update_layout(
            height=250 + (350 if gantt_display_possible else 0),
            title_text=f"{self.algorithm_name} 调度分析",
            title_x=0.5,
            font_family="SimHei, Noto Sans CJK SC, Microsoft YaHei, sans-serif",
            showlegend=False
        )

        final_fig.show()

# --- 其他调度器类保持不变 ---
# （FCFS, NonPreemptiveSJF, SRTF, NonPreemptivePriority, PreemptivePriority, RoundRobin, 
#   MultilevelQueueScheduler, MultilevelFeedbackQueueScheduler, 工具函数和主函数）
# 此处省略以保持简洁，实际使用时需包含完整代码

if __name__ == "__main__":
    main_cli()    