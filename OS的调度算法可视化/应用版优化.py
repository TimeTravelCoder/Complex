
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

# Global variable to store priority mode preference
PRIORITY_MODE_LOW_IS_HIGH: Optional[bool] = None

class Process:
    """进程类，包含进程的各项属性"""
    def __init__(self, pid: int, arrival_time: int, run_time: int, priority: Optional[int] = None):
        self.pid = pid
        self.arrival_time = arrival_time
        self.run_time = run_time
        self.priority = priority
        self._initial_run_time = run_time
        self._initial_priority = priority

        self.remaining_time = run_time
        self.start_time = None
        self.completion_time = None
        self.waiting_time = None
        self.turnaround_time = None
        self.response_time = None
        self.current_queue = 0

    def __str__(self):
        prio_str = 'N/A'
        if self.priority is not None:
            prio_str = str(self.priority)
            if PRIORITY_MODE_LOW_IS_HIGH is not None:
                 prio_str += " (L=H)" if PRIORITY_MODE_LOW_IS_HIGH else " (H=H)"
        return f"进程 {self.pid} (到达:{self.arrival_time}, 运行:{self.run_time}, 优先级:{prio_str})"

    def calculate_metrics(self):
        if self.completion_time is not None and self.start_time is not None:
            self.turnaround_time = self.completion_time - self.arrival_time
            self.waiting_time = self.turnaround_time - self.run_time
            if self.start_time is not None:
                 self.response_time = self.start_time - self.arrival_time
            else:
                self.response_time = None
        else:
            self.turnaround_time = None
            self.waiting_time = None
            self.response_time = None

    def reset(self):
        self.remaining_time = self._initial_run_time
        self.start_time = None
        self.completion_time = None
        self.waiting_time = None
        self.turnaround_time = None
        self.response_time = None
        self.current_queue = 0


class Scheduler(ABC):
    def __init__(self, processes: List[Process], name: str):
        self.processes = [Process(p.pid, p.arrival_time, p._initial_run_time, p._initial_priority) for p in processes]
        for p_obj in self.processes:
            p_obj.reset()

        self.ready_queue = []
        self.current_time = 0
        self.execution_sequence = []
        self.completed_processes = []
        self.current_process = None
        self.name = name
        self.config = {}
        if PRIORITY_MODE_LOW_IS_HIGH is not None and \
           ("priority" in name.lower() or "优先级" in name.lower() or "多级队列" in name.lower()):
            self.config["优先级模式"] = "数值越小优先级越高" if PRIORITY_MODE_LOW_IS_HIGH else "数值越大优先级越高"

    def update_ready_queue(self) -> None:
        for process in self.processes:
            add_to_ready = False
            if (process.arrival_time <= self.current_time and
                process not in self.completed_processes and
                process not in self.ready_queue and
                process != self.current_process):
                if process.remaining_time > 0:
                    add_to_ready = True
                elif process.run_time == 0 and process.start_time is None:
                    add_to_ready = True
            if add_to_ready:
                self.ready_queue.append(process)

    def execute_process(self, process: Process, duration: int) -> bool:
        if process.start_time is None:
            process.start_time = self.current_time

        actual_duration = min(duration, process.remaining_time)
        process.remaining_time -= actual_duration
        end_time = self.current_time + actual_duration
        completed = process.remaining_time <= 0

        status = "完成" if completed else "执行"
        if actual_duration > 0 or (actual_duration == 0 and completed):
            self.execution_sequence.append({
                "pid": process.pid, "start_time": self.current_time,
                "end_time": end_time, "status": status
            })

        self.current_time = end_time

        if completed:
            process.completion_time = end_time
            process.calculate_metrics()
            self.completed_processes.append(process)
            if process in self.ready_queue:
                self.ready_queue.remove(process)
            return True
        return False

    def is_preemptive(self) -> bool:
        return False

    @abstractmethod
    def select_process(self) -> Optional[Process]:
        pass

    def run(self) -> Dict:
        self.current_time = 0
        self.execution_sequence = []
        self.completed_processes = []
        self.current_process = None
        self.ready_queue = []
        for p_obj in self.processes:
            p_obj.reset()

        max_rt_sum = sum(p._initial_run_time for p in self.processes)
        max_at_sum = sum(p.arrival_time for p in self.processes)
        estimated_max_time = max_rt_sum + max_at_sum + len(self.processes) * 10 + 100
        if not self.processes: estimated_max_time = 100


        while len(self.completed_processes) < len(self.processes):
            self.update_ready_queue()

            if self.current_process and self.current_process.remaining_time > 0 and not self.is_preemptive():
                next_process_to_run = self.current_process
            else:
                next_process_to_run = self.select_process()

            if next_process_to_run:
                if self.is_preemptive() and self.current_process and \
                   self.current_process != next_process_to_run and \
                   self.current_process.remaining_time > 0:
                    self.handle_uncompleted_process(self.current_process)

                self.current_process = next_process_to_run
                if self.current_process in self.ready_queue:
                    self.ready_queue.remove(self.current_process)

                exec_slice = self.get_execution_time(self.current_process)
                completed = self.execute_process(self.current_process, exec_slice)

                if completed:
                    self.current_process = None
                elif self.is_preemptive():
                    self.handle_uncompleted_process(self.current_process)
                    self.current_process = None
            else:
                if not self.ready_queue and not self.current_process:
                    future_arrivals = [p.arrival_time for p in self.processes
                                       if p.arrival_time > self.current_time and
                                       (p.remaining_time > 0 or (p.run_time == 0 and p.start_time is None))]
                    if future_arrivals:
                        self.current_time = min(future_arrivals)
                        continue
                    else:
                        if all(p_obj in self.completed_processes or (p_obj.run_time == 0 and p_obj.start_time is not None) for p_obj in self.processes):
                            break
                        else:
                            self.current_time +=1
                else:
                    self.current_time += 1

            if self.current_time > estimated_max_time * 2 and estimated_max_time > 0 : # Failsafe
                 print(f"警告: 模拟时间 ({self.current_time}) 过长（预估 {estimated_max_time}）。可能中止。")
                 # for p_debug in self.processes:
                 #     if p_debug not in self.completed_processes: print(p_debug)
                 break
        return self.calculate_system_metrics()

    def get_execution_time(self, process: Process) -> int:
        if process.run_time == 0: return 0
        if self.is_preemptive(): return 1
        return process.remaining_time

    def handle_uncompleted_process(self, process: Process) -> None:
        if process.remaining_time > 0 and process not in self.ready_queue:
            self.ready_queue.append(process)

    def calculate_system_metrics(self) -> Dict:
        for p_obj in self.completed_processes:
            p_obj.calculate_metrics()

        if not self.completed_processes:
            return {"cpu_utilization": 0, "throughput": 0, "avg_turnaround_time": 0,
                    "avg_waiting_time": 0, "avg_response_time": 0}

        makespan = 0
        if self.completed_processes:
            makespan = max(p.completion_time for p in self.completed_processes if p.completion_time is not None)
        
        total_busy_time = sum(p._initial_run_time for p in self.completed_processes)
        cpu_util = total_busy_time / makespan if makespan > 0 else 0
        throughput_val = len(self.completed_processes) / makespan if makespan > 0 else 0

        avg_tt_sum = sum(p.turnaround_time for p in self.completed_processes if p.turnaround_time is not None)
        avg_wt_sum = sum(p.waiting_time for p in self.completed_processes if p.waiting_time is not None)
        avg_rt_sum = sum(p.response_time for p in self.completed_processes if p.response_time is not None)
        
        num_valid_metrics = len([p for p in self.completed_processes if p.turnaround_time is not None])
        if num_valid_metrics == 0:
            return {"cpu_utilization": cpu_util, "throughput": throughput_val,
                    "avg_turnaround_time": 0, "avg_waiting_time": 0, "avg_response_time": 0}

        return {
            "cpu_utilization": cpu_util, "throughput": throughput_val,
            "avg_turnaround_time": avg_tt_sum / num_valid_metrics,
            "avg_waiting_time": avg_wt_sum / num_valid_metrics,
            "avg_response_time": avg_rt_sum / num_valid_metrics
        }

    def print_results(self) -> None:
        clear_screen()
        print(f"\n===== {self.name} 调度算法 =====")
        if self.config:
            print("\n配置参数:")
            for key, value in self.config.items(): print(f"  {key}: {value}")

        print("\n输入进程信息 (原始):")
        for p_orig_data in sorted(self.processes, key=lambda p_item: p_item.pid) :
            prio_str = 'N/A'
            if p_orig_data._initial_priority is not None:
                prio_str = str(p_orig_data._initial_priority)
                if PRIORITY_MODE_LOW_IS_HIGH is not None:
                    prio_str += " (L=H)" if PRIORITY_MODE_LOW_IS_HIGH else " (H=H)"
            print(f"  进程 {p_orig_data.pid} (到达:{p_orig_data.arrival_time}, 运行:{p_orig_data._initial_run_time}, 优先级:{prio_str})")

        print("\n执行序列:")
        if not self.execution_sequence: print("  无执行活动。")
        else:
            for ex in self.execution_sequence:
                print(f"  时间 [{ex['start_time']}-{ex['end_time']}]: 进程 {ex['pid']} ({ex['status']})")

        print("\n进程性能指标:")
        headers = ["PID", "到达", "运行", "优先级", "完成", "周转", "等待", "响应"]
        col_widths = [5, 6, 6, 9, 6, 6, 6, 6]
        header_line = " | ".join([f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)])
        print("  " + header_line)
        print("  " + "-" * len(header_line))

        for p_result in sorted(self.processes, key=lambda p_item: p_item.pid):
            prio_val = p_result._initial_priority
            prio_disp_str = str(prio_val) if prio_val is not None else 'N/A'

            if p_result.completion_time is not None:
                row_items = [
                    p_result.pid, p_result.arrival_time, p_result._initial_run_time, prio_disp_str,
                    p_result.completion_time,
                    f"{p_result.turnaround_time:.2f}" if p_result.turnaround_time is not None else "N/A",
                    f"{p_result.waiting_time:.2f}" if p_result.waiting_time is not None else "N/A",
                    f"{p_result.response_time:.2f}" if p_result.response_time is not None else "N/A"
                ]
            else:
                row_items = [
                    p_result.pid, p_result.arrival_time, p_result._initial_run_time, prio_disp_str,
                    "未完成", "N/A", "N/A", "N/A"
                ]
            row_line = " | ".join([f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row_items)])
            print("  " + row_line)

        metrics = self.calculate_system_metrics()
        print("\n系统性能指标:")
        print(f"  CPU 利用率: {metrics['cpu_utilization']*100:.2f}%")
        print(f"  吞吐量: {metrics['throughput']:.2f} 进程/单位时间")
        print(f"  平均周转时间: {metrics['avg_turnaround_time']:.2f} 单位时间")
        print(f"  平均等待时间: {metrics['avg_waiting_time']:.2f} 单位时间")
        print(f"  平均响应时间: {metrics['avg_response_time']:.2f} 单位时间")

    def visualize(self) -> None:
        process_colors = {}
        sorted_scheduler_processes = sorted(self.processes, key=lambda p: p.pid)

        for i, process in enumerate(sorted_scheduler_processes):
            color_index = i % 10
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            process_colors[process.pid] = colors[color_index]

        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.7, 0.3],
            specs=[[{"type": "xy"}], [{"type": "table"}]],
            subplot_titles=(f"{self.name} 调度甘特图", "系统性能指标")
        )

        gantt_data = []
        for ex in self.execution_sequence:
            if ex["status"] != "抢占" and ex["end_time"] > ex["start_time"]:
                hover_text_content = f"进程 {ex['pid']}<br>时间: {ex['start_time']}-{ex['end_time']}<br>状态: {ex['status']}"
                gantt_data.append(dict(Task=f"进程 {ex['pid']}", Start=ex['start_time'], 
                                       Finish=ex['end_time'], PID=ex['pid'],
                                       HoverText=hover_text_content))

        all_pids_for_y_axis = sorted([p.pid for p in self.processes], reverse=True)
        y_category_order = [f"进程 {pid}" for pid in all_pids_for_y_axis]

        if gantt_data:
            df = pd.DataFrame(gantt_data)
            for pid_val in sorted(list(df['PID'].unique())):
                task_name = f"进程 {pid_val}"
                df_pid = df[df['PID'] == pid_val]
                fig.add_trace(
                    go.Bar(
                        y=df_pid['Task'], x=df_pid['Finish'] - df_pid['Start'],
                        base=df_pid['Start'], orientation='h',
                        marker_color=process_colors.get(pid_val, '#000000'), name=task_name,
                        hovertext=df_pid['HoverText'],
                        hoverinfo="text",  # CORRECTED HERE
                        showlegend=False
                    ), row=1, col=1
                )
            legend_pids = set()
            for pid_val in sorted(list(df['PID'].unique())):
                if pid_val not in legend_pids:
                    fig.add_trace(go.Bar(y=[f"进程 {pid_val}"], x=[0],showlegend=True,
                                         marker_color=process_colors.get(pid_val, '#000000'), name=f"进程 {pid_val}"), row=1,col=1)
                    legend_pids.add(pid_val)

        metrics_table = self.calculate_system_metrics()
        fig.add_trace(
            go.Table(
                 header=dict(values=['指标', '值'], font=dict(size=14, family='SimHei, Arial'), align='center'),
                cells=dict(
                    values=[
                        ['CPU 利用率', '吞吐量', '平均周转时间', '平均等待时间', '平均响应时间'],
                        [f"{metrics_table['cpu_utilization']*100:.2f}%",
                         f"{metrics_table['throughput']:.2f} 进程/单位时间",
                         f"{metrics_table['avg_turnaround_time']:.2f} 单位时间",
                         f"{metrics_table['avg_waiting_time']:.2f} 单位时间",
                         f"{metrics_table['avg_response_time']:.2f} 单位时间"]
                    ],
                    font=dict(size=12, family='SimHei, Arial'), align='center'
                )
            ), row=2, col=1
        )

        fig.update_layout(
            title_text=f"{self.name} 调度算法可视化",
            title_font=dict(size=20, family='SimHei, Arial'),
            barmode='stack', height=max(600, 100 + len(self.processes) * 30 + 200),
            font=dict(family='SimHei, Arial'), legend_title_text='图例',
            yaxis1=dict(categoryorder='array', categoryarray=y_category_order, title_text='进程', titlefont=dict(family='SimHei, Arial')),
            xaxis1=dict(title_text='时间', titlefont=dict(family='SimHei, Arial'), rangemode='tozero', separatethousands=False)
        )
        fig.show()


class FCFSScheduler(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "先来先服务(FCFS)")
    def select_process(self) -> Optional[Process]:
        if not self.ready_queue: return None
        return sorted(self.ready_queue, key=lambda p: (p.arrival_time, p.pid))[0]
    def get_execution_time(self, process: Process) -> int:
        if process.run_time == 0: return 0
        return process.remaining_time

class NonPreemptiveSJFScheduler(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "非抢占式短作业优先(SJF)")
    def select_process(self) -> Optional[Process]:
        if not self.ready_queue: return None
        return sorted(self.ready_queue, key=lambda p: (p._initial_run_time, p.arrival_time, p.pid))[0]
    def get_execution_time(self, process: Process) -> int:
        if process.run_time == 0: return 0
        return process.remaining_time

class PreemptiveSJFScheduler(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "抢占式短作业优先(SRTF)")
    def is_preemptive(self) -> bool: return True
    def select_process(self) -> Optional[Process]:
        candidates = self.ready_queue[:]
        if self.current_process and self.current_process.remaining_time > 0 and self.current_process not in candidates:
            candidates.append(self.current_process)
        if not candidates: return None
        return sorted(candidates, key=lambda p: (p.remaining_time, p.arrival_time, p.pid))[0]
    def get_execution_time(self, process: Process) -> int:
        if process.run_time == 0: return 0
        return 1

class NonPreemptivePriorityScheduler(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "非抢占式优先级")
        self._check_priorities()
    def _check_priorities(self):
        if not all(p._initial_priority is not None for p in self.processes):
            raise ValueError("所有进程必须有优先级。")
    def select_process(self) -> Optional[Process]:
        if not self.ready_queue: return None
        if PRIORITY_MODE_LOW_IS_HIGH:
            return sorted(self.ready_queue, key=lambda p: (p._initial_priority, p.arrival_time, p.pid))[0]
        else:
            return sorted(self.ready_queue, key=lambda p: (-p._initial_priority, p.arrival_time, p.pid))[0]
    def get_execution_time(self, process: Process) -> int:
        if process.run_time == 0: return 0
        return process.remaining_time

class PreemptivePriorityScheduler(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes, "抢占式优先级")
        self._check_priorities()
    def _check_priorities(self):
        if not all(p._initial_priority is not None for p in self.processes):
            raise ValueError("所有进程必须有优先级。")
    def is_preemptive(self) -> bool: return True
    def select_process(self) -> Optional[Process]:
        candidates = self.ready_queue[:]
        if self.current_process and self.current_process.remaining_time > 0 and self.current_process not in candidates:
            candidates.append(self.current_process)
        if not candidates: return None
        if PRIORITY_MODE_LOW_IS_HIGH:
            return sorted(candidates, key=lambda p: (p._initial_priority, p.arrival_time, p.pid))[0]
        else:
            return sorted(candidates, key=lambda p: (-p._initial_priority, p.arrival_time, p.pid))[0]
    def get_execution_time(self, process: Process) -> int:
        if process.run_time == 0: return 0
        return 1

class RoundRobinScheduler(Scheduler):
    def __init__(self, processes: List[Process], time_quantum: int):
        super().__init__(processes, "时间片轮转(RR)")
        if time_quantum <= 0: raise ValueError("时间片必须为正。")
        self.time_quantum = time_quantum
        self.config["时间片大小"] = time_quantum
    def is_preemptive(self) -> bool: return True
    def select_process(self) -> Optional[Process]:
        if not self.ready_queue: return None
        return self.ready_queue[0]
    def get_execution_time(self, process: Process) -> int:
        if process.run_time == 0: return 0
        return min(self.time_quantum, process.remaining_time)
    def handle_uncompleted_process(self, process: Process):
        if process.remaining_time > 0 and process not in self.ready_queue:
            self.ready_queue.append(process)

class MultiLevelQueueScheduler(Scheduler):
    def __init__(self, processes: List[Process], queue_configs: List[Dict]):
        super().__init__(processes, "多级队列调度")
        self.queue_configs = queue_configs
        self.queues: List[List[Process]] = [[] for _ in range(len(queue_configs))]
        self.config["队列数量"] = len(queue_configs)
        self.config["队列配置"] = []
        for i, qc in enumerate(self.queue_configs):
            cfg_str = f"队列{i} ({qc['algorithm']}"
            if qc['algorithm'] == "RR": cfg_str += f", Q={qc.get('time_quantum', 'N/A')}"
            cfg_str += ")"
            self.config["队列配置"].append(cfg_str)
            if qc['algorithm'] not in ["FCFS", "SJF", "Priority", "RR"]:
                raise ValueError(f"队列 {i} 算法 '{qc['algorithm']}' 不支持.")
            if qc['algorithm'] == "RR" and (qc.get('time_quantum', 0) <= 0):
                raise ValueError(f"队列 {i} RR时间片必须为正.")
        for p in self.processes:
            prio = p._initial_priority
            if prio is not None and 0 <= prio < len(self.queues):
                p.current_queue = prio
            else:
                p.current_queue = 0

    def update_ready_queue(self):
        for p in self.processes:
            is_in_any_mlq_queue = any(p in q_list for q_list in self.queues)
            add_to_queue = False
            if p.arrival_time <= self.current_time and \
               p not in self.completed_processes and \
               p != self.current_process and \
               not is_in_any_mlq_queue:
                if p.remaining_time > 0 or (p.run_time == 0 and p.start_time is None):
                    add_to_queue = True
            if add_to_queue:
                q_idx = p.current_queue
                self.queues[q_idx].append(p)
                if self.queue_configs[q_idx]['algorithm'] == 'FCFS':
                    self.queues[q_idx].sort(key=lambda pi: (pi.arrival_time, pi.pid))
        self.ready_queue = [item for q in self.queues for item in q]

    def select_process(self) -> Optional[Process]:
        for i, q_list in enumerate(self.queues):
            if not q_list: continue
            algo = self.queue_configs[i]['algorithm']
            if algo == "FCFS": return q_list[0]
            elif algo == "SJF":
                return sorted(q_list, key=lambda p: (p._initial_run_time, p.arrival_time, p.pid))[0]
            elif algo == "Priority":
                # Ensure _initial_priority is not None for processes in a Priority queue
                # This should be caught by validate_processes or specific checks if a process lands here without priority
                if PRIORITY_MODE_LOW_IS_HIGH:
                    return sorted(q_list, key=lambda p: (p._initial_priority if p._initial_priority is not None else float('inf'), p.arrival_time, p.pid))[0]
                else:
                    return sorted(q_list, key=lambda p: (-(p._initial_priority if p._initial_priority is not None else float('-inf')), p.arrival_time, p.pid))[0]
            elif algo == "RR": return q_list[0]
        return None

    def is_preemptive(self) -> bool:
        if self.current_process:
            curr_q_idx = self.current_process.current_queue
            for i in range(curr_q_idx):
                if self.queues[i]: return True
            if self.queue_configs[curr_q_idx]['algorithm'] == "RR": return True
        return False

    def get_execution_time(self, process: Process) -> int:
        if process.run_time == 0: return 0
        q_idx = process.current_queue
        algo = self.queue_configs[q_idx]['algorithm']
        if algo == "RR":
            return min(self.queue_configs[q_idx]['time_quantum'], process.remaining_time)
        return 1

    def handle_uncompleted_process(self, process: Process):
        if process.remaining_time > 0:
            q_idx = process.current_queue
            if process not in self.queues[q_idx]:
                if self.queue_configs[q_idx]['algorithm'] == 'RR':
                    self.queues[q_idx].append(process)
                else:
                    self.queues[q_idx].insert(0, process)
                    if self.queue_configs[q_idx]['algorithm'] == 'FCFS':
                         self.queues[q_idx].sort(key=lambda pi: (pi.arrival_time, pi.pid))

    def execute_process(self, process: Process, duration: int) -> bool:
        q_idx = process.current_queue
        if process in self.queues[q_idx]:
            self.queues[q_idx].remove(process)
        return super().execute_process(process, duration)

class MultilevelFeedbackQueueScheduler(Scheduler):
    def __init__(self, processes: List[Process], queue_count: int, time_quantums: List[int],
                 enable_aging: bool = False, aging_threshold: int = 10):
        super().__init__(processes, "多级反馈队列")
        if not (queue_count >= 1 and len(time_quantums) == queue_count):
             raise ValueError("队列数量和时间片列表长度必须匹配且 >=1.")
        if queue_count > 1:
            for i in range(queue_count -1 ):
                if time_quantums[i] <=0: raise ValueError(f"MLFQ RR队列 {i} 时间片必须为正.")
        if enable_aging and aging_threshold <=0: raise ValueError("老化阈值必须为正.")

        self.queue_count = queue_count
        self.time_quantums = time_quantums
        self.queues: List[List[Process]] = [[] for _ in range(queue_count)]
        self.enable_aging = enable_aging
        self.aging_threshold = aging_threshold if enable_aging else float('inf')
        self.process_wait_in_queue_time: Dict[int, int] = {p.pid: 0 for p in self.processes}

        self.config["队列数量"] = queue_count
        tq_disp = [ (str(tq) if i < queue_count -1 or queue_count == 1 and tq != sys.maxsize else "FCFS") for i,tq in enumerate(time_quantums)]
        if queue_count == 1 and time_quantums[0] == sys.maxsize : tq_disp = ["FCFS"] # Special case: 1 queue that is FCFS
        self.config["各队列策略/时间片"] = ", ".join(tq_disp)
        self.config["启用老化"] = "是" if enable_aging else "否"
        if enable_aging: self.config["老化阈值"] = aging_threshold
        for p in self.processes: p.current_queue = 0

    def update_ready_queue(self):
        for p in self.processes:
            is_in_any_mlfq_queue = any(p in q_list for q_list in self.queues)
            add_to_q0 = False
            if p.arrival_time <= self.current_time and \
               p not in self.completed_processes and \
               p != self.current_process and \
               not is_in_any_mlfq_queue:
                if p.remaining_time > 0 or (p.run_time == 0 and p.start_time is None):
                    add_to_q0 = True
            if add_to_q0:
                p.current_queue = 0
                self.queues[0].append(p)
                self.process_wait_in_queue_time[p.pid] = 0
        if self.enable_aging: self._apply_aging()
        self.ready_queue = [item for q in self.queues for item in q]

    def _apply_aging(self):
        for q_idx_from in range(self.queue_count - 1, 0, -1):
            for p_wait in self.queues[q_idx_from]:
                if self.current_process != p_wait:
                    self.process_wait_in_queue_time[p_wait.pid] = self.process_wait_in_queue_time.get(p_wait.pid, 0) + 1
            moved_this_cycle = []
            for p_candidate in self.queues[q_idx_from]:
                if self.process_wait_in_queue_time[p_candidate.pid] >= self.aging_threshold:
                    target_q_idx = q_idx_from - 1
                    p_candidate.current_queue = target_q_idx
                    self.queues[target_q_idx].append(p_candidate)
                    moved_this_cycle.append(p_candidate)
                    self.process_wait_in_queue_time[p_candidate.pid] = 0
            for p_moved in moved_this_cycle:
                self.queues[q_idx_from].remove(p_moved)

    def select_process(self) -> Optional[Process]:
        for i, q_list in enumerate(self.queues):
            if q_list:
                if i == self.queue_count - 1:
                    return sorted(q_list, key=lambda p: (p.arrival_time, p.pid))[0]
                else: return q_list[0]
        return None

    def is_preemptive(self) -> bool: return True

    def get_execution_time(self, process: Process) -> int:
        if process.run_time == 0: return 0
        q_idx = process.current_queue
        if q_idx == self.queue_count - 1:
            return process.remaining_time
        return min(self.time_quantums[q_idx], process.remaining_time)

    def handle_uncompleted_process(self, process: Process):
        if process.remaining_time > 0:
            prev_q_idx = process.current_queue
            self.process_wait_in_queue_time[process.pid] = 0

            if prev_q_idx < self.queue_count - 1:
                next_q_idx = prev_q_idx + 1
                process.current_queue = next_q_idx
                self.queues[next_q_idx].append(process)
            else:
                self.queues[prev_q_idx].append(process)
                self.queues[prev_q_idx].sort(key=lambda p: (p.arrival_time, p.pid))

    def execute_process(self, process: Process, duration: int) -> bool:
        q_idx = process.current_queue
        if process in self.queues[q_idx]:
            self.queues[q_idx].remove(process)
        self.process_wait_in_queue_time[process.pid] = 0
        return super().execute_process(process, duration)

# --- Auxiliary functions ---
def clear_screen():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def get_priority_mode_preference_from_user():
    global PRIORITY_MODE_LOW_IS_HIGH
    print("\n请选择优先级模式:")
    print("1. 数值越小，优先级越高 (例如：0是最高优先级)")
    print("2. 数值越大，优先级越高 (例如：10比0优先级高)")
    while True:
        choice = get_integer_input("输入选择 (1/2): ", 1, 2)
        if choice == 1: PRIORITY_MODE_LOW_IS_HIGH = True; print("已设置为: 数值越小，优先级越高。"); break
        elif choice == 2: PRIORITY_MODE_LOW_IS_HIGH = False; print("已设置为: 数值越大，优先级越高。"); break

def validate_processes(processes: List[Process], algorithm_name: str) -> bool:
    global PRIORITY_MODE_LOW_IS_HIGH
    needs_priority_defined = False
    if "priority" in algorithm_name.lower() or "优先级" in algorithm_name: needs_priority_defined = True
    if "多级队列" in algorithm_name and "反馈" not in algorithm_name: needs_priority_defined = True

    if needs_priority_defined:
        if not all(p._initial_priority is not None for p in processes):
            print(f"错误: 算法 '{algorithm_name}' 要求所有进程都定义优先级值。")
            return False
        if PRIORITY_MODE_LOW_IS_HIGH is None:
            print(f"注意: 算法 '{algorithm_name}' 使用优先级。")
            get_priority_mode_preference_from_user()
    return True

def get_integer_input(prompt: str, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    while True:
        try:
            val_str = input(prompt).strip()
            if not val_str: print("输入不能为空。"); continue
            val = int(val_str)
            if (min_value is not None and val < min_value) or \
               (max_value is not None and val > max_value):
                err_parts = []
                if min_value is not None: err_parts.append(f"不小于 {min_value}")
                if max_value is not None: err_parts.append(f"不大于 {max_value}")
                print(f"输入超出范围。请输入{'且'.join(err_parts)}的整数。")
            else: return val
        except ValueError: print("无效输入，请输入整数。")

def get_yes_no_input(prompt: str) -> bool:
    while True:
        res = input(prompt + " (y/n): ").strip().lower()
        if res in ['y', 'yes', '是']: return True
        if res in ['n', 'no', '否']: return False
        print("无效输入。请输入 'y' (是) 或 'n' (否).")

ProcessDataType = List[Tuple[int, int, int, Optional[int]]]

def manual_input_processes_data() -> ProcessDataType:
    global PRIORITY_MODE_LOW_IS_HIGH
    process_params_list: ProcessDataType = []
    count = get_integer_input("请输入进程数量 (1-100): ", 1, 100)
    has_priority_input = get_yes_no_input("是否为进程设置优先级?")
    if has_priority_input and PRIORITY_MODE_LOW_IS_HIGH is None:
        get_priority_mode_preference_from_user()
    for i in range(count):
        pid = i + 1; print(f"\n--- 输入进程 {pid} ---")
        arrival = get_integer_input(f"  到达时间 (>=0): ", 0)
        burst = get_integer_input(f"  运行时间 (>=0): ", 0)
        priority_val = get_integer_input(f"  优先级值 (整数): ") if has_priority_input else None
        process_params_list.append((pid, arrival, burst, priority_val))
    return process_params_list

def random_generate_processes_data() -> ProcessDataType:
    global PRIORITY_MODE_LOW_IS_HIGH
    process_params_list: ProcessDataType = []
    count = get_integer_input("随机生成进程数量 (1-100): ", 1, 100)
    max_arrival = get_integer_input("最大到达时间 (>=0): ", 0)
    max_burst = get_integer_input("最大运行时间 (>=0): ", 0)
    has_priority_random = get_yes_no_input("是否随机生成优先级?")
    min_prio_val, max_prio_val = 0, 9
    if has_priority_random and PRIORITY_MODE_LOW_IS_HIGH is None:
        get_priority_mode_preference_from_user()
    for i in range(count):
        pid = i + 1; arrival = random.randint(0, max_arrival)
        burst = random.randint(0, max_burst)
        priority_val = random.randint(min_prio_val, max_prio_val) if has_priority_random else None
        process_params_list.append((pid, arrival, burst, priority_val))
    return process_params_list

def create_processes_from_data(data: ProcessDataType) -> List[Process]:
    return [Process(pid, at, rt, prio) for pid, at, rt, prio in data]

def run_scheduling_simulation():
    global PRIORITY_MODE_LOW_IS_HIGH
    original_process_data: Optional[ProcessDataType] = None

    while True:
        clear_screen()
        if original_process_data is None:
            PRIORITY_MODE_LOW_IS_HIGH = None
            print("====== 操作系统调度算法模拟器 ======\n")
            print("1. 手动输入进程数据"); print("2. 随机生成进程数据"); print("0. 退出程序")
            choice = get_integer_input("请选择操作: ", 0, 2)
            if choice == 0: break
            elif choice == 1: original_process_data = manual_input_processes_data()
            else: original_process_data = random_generate_processes_data()
        if not original_process_data:
            print("错误: 未能加载进程数据。"); original_process_data = None; input("按回车键返回..."); continue

        while True:
            clear_screen(); print("当前进程数据已加载。选择调度算法或操作:\n")
            temp_procs_disp = create_processes_from_data(original_process_data)
            print("当前进程:")
            for p_d in sorted(temp_procs_disp,key=lambda x:x.pid): print(f"  {p_d}")
            if PRIORITY_MODE_LOW_IS_HIGH is not None:
                 print(f"当前优先级模式: {'数值越小优先级越高' if PRIORITY_MODE_LOW_IS_HIGH else '数值越大优先级越高'}")
            print("-" * 30)
            print("1. FCFS  2. SJF (非抢占)  3. SRTF (抢占SJF)")
            print("4. 优先级 (非抢占)      5. 优先级 (抢占)")
            print("6. 时间片轮转 (RR)")
            print("7. 多级队列调度"); print("8. 多级反馈队列调度")
            print("\n9. 重新输入/生成进程数据")
            can_change_prio_mode = PRIORITY_MODE_LOW_IS_HIGH is not None or \
                                   any(p[3] is not None for p in original_process_data)
            if can_change_prio_mode: print("10. 更改当前数据集的优先级解释模式")
            print("0. 退出程序")

            algo_choice = get_integer_input("选择 (0-10): ", 0, 10 if can_change_prio_mode else 9)

            if algo_choice == 0: print("感谢使用！"); return
            if algo_choice == 9: original_process_data = None; break
            if algo_choice == 10 and can_change_prio_mode:
                get_priority_mode_preference_from_user(); continue
            elif algo_choice == 10 and not can_change_prio_mode: # Should not happen if input constrained
                print("无效选择。"); input("按回车键..."); continue


            current_scheduler_processes = create_processes_from_data(original_process_data)
            scheduler: Optional[Scheduler] = None
            algo_name_map = {1: "FCFS", 2: "SJF (非抢占)", 3: "SRTF", 4: "优先级 (非抢占)",
                             5: "优先级 (抢占)", 6: "RR", 7: "多级队列调度", 8: "多级反馈队列"}
            current_algo_name = algo_name_map.get(algo_choice, "未知算法")

            if not validate_processes(current_scheduler_processes, current_algo_name):
                input("按回车键继续..."); continue
            
            try:
                if algo_choice == 1: scheduler = FCFSScheduler(current_scheduler_processes)
                elif algo_choice == 2: scheduler = NonPreemptiveSJFScheduler(current_scheduler_processes)
                elif algo_choice == 3: scheduler = PreemptiveSJFScheduler(current_scheduler_processes)
                elif algo_choice == 4: scheduler = NonPreemptivePriorityScheduler(current_scheduler_processes)
                elif algo_choice == 5: scheduler = PreemptivePriorityScheduler(current_scheduler_processes)
                elif algo_choice == 6:
                    tq = get_integer_input("RR 时间片大小 (>=1): ", 1)
                    scheduler = RoundRobinScheduler(current_scheduler_processes, tq)
                elif algo_choice == 7:
                    print("\n--- 配置多级队列 ---"); num_q = get_integer_input("队列数量 (1-5): ", 1, 5); q_cfgs = []
                    print("进程的 _initial_priority (0, 1, ...) 决定其初始队列。")
                    for i in range(num_q):
                        print(f"\n--- 队列 {i} (优先级 {i}) ---")
                        print("  1.FCFS  2.SJF(队内非抢占)  3.Priority(队内非抢占)  4.RR")
                        q_algo = get_integer_input(f"  队列 {i} 算法 (1-4): ", 1, 4); cfg = {}
                        if q_algo == 1: cfg['algorithm'] = "FCFS"
                        elif q_algo == 2: cfg['algorithm'] = "SJF"
                        elif q_algo == 3: cfg['algorithm'] = "Priority"
                        elif q_algo == 4: cfg['algorithm'] = "RR"; cfg['time_quantum'] = get_integer_input(f"  队列 {i} RR 时间片 (>=1): ", 1)
                        q_cfgs.append(cfg)
                    scheduler = MultiLevelQueueScheduler(current_scheduler_processes, q_cfgs)
                elif algo_choice == 8:
                    print("\n--- 配置多级反馈队列 ---"); num_fbq = get_integer_input("队列数量 (>=1, 最后一个为FCFS): ", 1, 5); fb_tqs = []
                    for i in range(num_fbq):
                        if i < num_fbq - 1 or (num_fbq == 1 and get_yes_no_input(f"队列0是否为RR (否则为FCFS)?")): # Allow single queue MLFQ to be RR or FCFS
                            fb_tqs.append(get_integer_input(f"  队列 {i} (RR) 时间片 (>=1): ", 1))
                        else: print(f"  队列 {i} 为 FCFS。"); fb_tqs.append(sys.maxsize)
                    aging = get_yes_no_input("启用老化机制?"); ag_th = 10
                    if aging: ag_th = get_integer_input("老化阈值 (>=1): ", 1)
                    scheduler = MultilevelFeedbackQueueScheduler(current_scheduler_processes, num_fbq, fb_tqs, aging, ag_th)

                if scheduler:
                    print(f"\n运行 {scheduler.name}..."); scheduler.run(); scheduler.print_results()
                    if get_yes_no_input("\n显示可视化甘特图?"):
                        try: scheduler.visualize()
                        except Exception as e_vis: print(f"可视化错误: {e_vis}")
                elif algo_choice in algo_name_map:
                     print("调度器未能初始化。请检查进程数据或配置。")
            except ValueError as ve: print(f"配置或输入错误: {ve}")
            except Exception as e_run:
                print(f"调度运行时意外错误: {e_run}"); import traceback; traceback.print_exc()
            input("\n按回车键继续...");

if __name__ == "__main__":
    run_scheduling_simulation()
