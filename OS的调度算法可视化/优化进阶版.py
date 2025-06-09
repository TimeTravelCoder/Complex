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
import copy # For deep copying processes

# Global variable to store priority logic, set by user at runtime
# False: smaller value = higher priority (e.g., 0 is best)
# True: larger value = higher priority (e.g., 10 is best)
# This will be set in run_scheduling_simulation
PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER = False

class Process:
    """进程类，包含进程的各项属性"""
    def __init__(self, pid: int, arrival_time: int, run_time: int, priority: Optional[int] = None):
        self.pid = pid
        # Definitional attributes (should not change after creation for a given process set)
        self.arrival_time = arrival_time
        self.run_time = run_time  # This is the total required burst time
        self.priority = priority

        # State attributes (reset for each simulation run)
        self.remaining_time = run_time
        self.start_time = None  # 首次执行的时间
        self.completion_time = None  # 完成时间
        self.waiting_time = None  # 等待时间 = 周转时间 - 运行时间
        self.turnaround_time = None  # 周转时间 = 完成时间 - 到达时间
        self.response_time = None  # 响应时间 = 首次执行时间 - 到达时间
        self.current_queue = 0  # 用于多级队列和多级反馈队列

    def __str__(self):
        priority_val = self.priority if self.priority is not None else 'N/A'
        return f"进程 {self.pid} (到达:{self.arrival_time}, 运行:{self.run_time}, 优先级:{priority_val})"

    def calculate_metrics(self):
        """计算进程的性能指标"""
        if self.completion_time is not None and self.arrival_time is not None: # Ensure arrival_time is not None
            self.turnaround_time = self.completion_time - self.arrival_time
            if self.run_time is not None: # Ensure run_time is not None
                 self.waiting_time = self.turnaround_time - self.run_time
            if self.start_time is not None:
                 self.response_time = self.start_time - self.arrival_time


class Scheduler(ABC):
    """调度器基类"""
    def __init__(self, processes: List[Process], name: str, priority_logic_higher_is_better: bool = False):
        self.processes = processes # Schedulers should use this list of Process objects
        self.ready_queue = []  # 就绪队列
        self.current_time = 0  # 当前时间
        self.execution_sequence = []  # 执行序列记录
        self.completed_processes = []  # 已完成的进程
        self.current_process = None  # 当前正在运行的进程
        self.name = name  # 调度算法名称
        self.config = {}  # 调度算法配置参数
        self.priority_logic_higher_is_better = priority_logic_higher_is_better # Store priority logic

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
        if process.start_time is None:
            process.start_time = self.current_time

        actual_duration = min(duration, process.remaining_time) # Ensure we don't over-execute
        process.remaining_time -= actual_duration
        end_time = self.current_time + actual_duration
        completed = process.remaining_time <= 0

        status = "完成" if completed else "执行"
        self.execution_sequence.append({
            "pid": process.pid,
            "start_time": self.current_time,
            "end_time": end_time,
            "status": status
        })

        self.current_time = end_time

        if completed:
            process.completion_time = end_time
            process.calculate_metrics()
            self.completed_processes.append(process)
            if process in self.ready_queue: # Should have been removed before execution
                self.ready_queue.remove(process)
            return True
        return False

    def is_preemptive(self) -> bool:
        return False

    @abstractmethod
    def select_process(self) -> Optional[Process]:
        pass

    def run(self) -> Dict:
        while len(self.completed_processes) < len(self.processes):
            self.update_ready_queue()

            if not self.ready_queue and not self.current_process:
                # If no process is ready and no process is running, advance time to next arrival
                pending_processes = [p for p in self.processes if p not in self.completed_processes and p.arrival_time > self.current_time]
                if not pending_processes:
                    # All processes arrived or completed. If ready queue is still empty, something is wrong or done.
                    if not any(p.remaining_time > 0 for p in self.processes): # All processes truly done
                        break
                    # This case (no ready, no current, but pending processes exist far in future)
                    # suggests we might need to advance time if idle.
                    # Fallback: if stuck, increment time. Better: advance to next arrival.
                    if not self.execution_sequence or self.execution_sequence[-1]['end_time'] < self.current_time:
                        # If idle, advance time
                        next_arrival_times = [p.arrival_time for p in self.processes if p not in self.completed_processes and p.arrival_time > self.current_time]
                        if next_arrival_times:
                            self.current_time = min(next_arrival_times)
                            continue
                        elif self.current_process is None and not self.ready_queue: # No more processes to arrive
                             break 

                else:
                    self.current_time = min(p.arrival_time for p in pending_processes)
                continue # Re-evaluate after time advance


            next_process_to_run = self.select_process()

            if self.is_preemptive() and self.current_process and self.current_process.remaining_time > 0:
                if next_process_to_run != self.current_process and next_process_to_run is not None: # Preemption condition
                    # Preempt current_process
                    self.execution_sequence.append({
                        "pid": self.current_process.pid,
                        "start_time": self.current_time, # Mark preemption at current time
                        "end_time": self.current_time,
                        "status": "抢占"
                    })
                    self.handle_uncompleted_process(self.current_process, 0) # 0 executed duration for this step
                    self.current_process = None # Important: current_process is now None before picking new one

            if self.current_process is None: # If no process is running (or was just preempted)
                self.current_process = next_process_to_run
                if self.current_process:
                    # Remove from its specific ready queue structure
                    if isinstance(self, (MultiLevelQueueScheduler, MultilevelFeedbackQueueScheduler)):
                        if self.current_process in self.queues[self.current_process.current_queue]:
                            self.queues[self.current_process.current_queue].remove(self.current_process)
                    elif self.current_process in self.ready_queue:
                         self.ready_queue.remove(self.current_process)
            elif not self.is_preemptive() and self.current_process.remaining_time <= 0 : # Non-preemptive and finished
                self.current_process = None # Allow selecting a new process
                # Try to select again in the same time slice if one is ready
                self.current_process = next_process_to_run 
                if self.current_process: # If new one selected
                    if isinstance(self, (MultiLevelQueueScheduler, MultilevelFeedbackQueueScheduler)):
                        if self.current_process in self.queues[self.current_process.current_queue]:
                            self.queues[self.current_process.current_queue].remove(self.current_process)
                    elif self.current_process in self.ready_queue:
                         self.ready_queue.remove(self.current_process)


            if self.current_process:
                execution_duration_for_step = self.get_execution_time(self.current_process)
                completed = self.execute_process(self.current_process, execution_duration_for_step)

                if completed:
                    self.current_process = None
                elif self.is_preemptive(): # Preemptive and not completed
                    self.handle_uncompleted_process(self.current_process, execution_duration_for_step)
                    self.current_process = None # Allow re-selection in next iteration
                # If non-preemptive and not completed, it continues implicitly by remaining self.current_process
            
            else: # No process to run, advance time by 1 if ready queue was also empty
                if not self.ready_queue:
                    # Advance time to the next event (arrival) or increment if no arrivals soon
                    future_arrivals = [p.arrival_time for p in self.processes if p.arrival_time > self.current_time and p not in self.completed_processes]
                    if future_arrivals:
                        next_event_time = min(future_arrivals)
                        # Add idle time to execution sequence
                        if not self.execution_sequence or self.execution_sequence[-1]['end_time'] < next_event_time :
                             if self.current_time < next_event_time: # only if there's a gap
                                self.execution_sequence.append({
                                    "pid": "空闲", 
                                    "start_time": self.current_time, 
                                    "end_time": next_event_time, 
                                    "status": "空闲"
                                })
                        self.current_time = next_event_time
                    else: # No future arrivals, but some processes might still be in ready queue (handled by update_ready_queue) or running
                        if len(self.completed_processes) < len(self.processes): # still work to do
                             # This case should ideally be covered by current_process logic or ready_queue check
                             # If truly idle and unfinished work, small time increment to allow re-evaluation.
                             self.current_time +=1 
                        else: # All processes completed
                            break 
                # If ready queue is not empty, loop will call update_ready_queue and select_process

        return self.calculate_system_metrics()


    def get_execution_time(self, process: Process) -> int:
        return process.remaining_time

    def handle_uncompleted_process(self, process: Process, executed_duration: int) -> None:
        if process.remaining_time > 0: # Process is not finished
            # Default behavior: add back to the generic ready_queue if not already there.
            # Specific schedulers (RR, MFQS) will override this.
            if process not in self.ready_queue:
                self.ready_queue.append(process)


    def _get_sort_key_for_priority(self, process: Process) -> Tuple:
        # Helper for sorting by priority, respecting PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER
        # Lower effective_priority is better for min-sorting.
        if process.priority is None: # Should be caught by _check_priorities
            effective_priority = float('inf') 
        else:
            effective_priority = -process.priority if self.priority_logic_higher_is_better else process.priority
        return (effective_priority, process.arrival_time, process.pid)


    def calculate_system_metrics(self) -> Dict:
        for process in self.completed_processes: # Ensure all completed ones have metrics
            if process.turnaround_time is None: # Recalculate if somehow missed
                process.calculate_metrics()
        
        if not self.completed_processes:
            return {
                "cpu_utilization": 0, "throughput": 0,
                "avg_turnaround_time": 0, "avg_waiting_time": 0, "avg_response_time": 0
            }
        
        total_completion_time = max([p.completion_time for p in self.completed_processes if p.completion_time is not None], default=0)
        if total_completion_time == 0 and self.execution_sequence: # If only idle time was logged
            total_completion_time = max(ex['end_time'] for ex in self.execution_sequence) if self.execution_sequence else 0

        if total_completion_time == 0: # Avoid division by zero
             return {
                "cpu_utilization": 0, "throughput": 0,
                "avg_turnaround_time": 0, "avg_waiting_time": 0, "avg_response_time": 0
            }

        cpu_busy_time = sum([ex["end_time"] - ex["start_time"] for ex in self.execution_sequence if ex["pid"] != "空闲" and ex["status"] != "抢占"])
        
        cpu_utilization = cpu_busy_time / total_completion_time if total_completion_time > 0 else 0
        throughput = len(self.completed_processes) / total_completion_time if total_completion_time > 0 else 0
        
        avg_turnaround_time = sum([p.turnaround_time for p in self.completed_processes if p.turnaround_time is not None]) / len(self.completed_processes)
        avg_waiting_time = sum([p.waiting_time for p in self.completed_processes if p.waiting_time is not None]) / len(self.completed_processes)
        # Filter out processes that might not have a response time if they never started (should not happen for completed ones)
        valid_response_times = [p.response_time for p in self.completed_processes if p.response_time is not None]
        avg_response_time = sum(valid_response_times) / len(valid_response_times) if valid_response_times else 0
        
        return {
            "cpu_utilization": cpu_utilization, "throughput": throughput,
            "avg_turnaround_time": avg_turnaround_time, "avg_waiting_time": avg_waiting_time,
            "avg_response_time": avg_response_time
        }

    def print_results(self) -> None:
        clear_screen()
        print(f"\n===== {self.name} 调度算法 =====")
        if self.config:
            print("\n配置参数:")
            for key, value in self.config.items():
                print(f"  {key}: {value}")
        
        # Original processes info
        print("\n初始进程信息:")
        # Sort original_processes by PID for consistent display if they were part of the input to __init__
        # However, self.processes are the actual objects used and potentially modified.
        # For clarity, iterate through sorted list of PIDs from the original set.
        
        # Create a dictionary of processes by PID for easy lookup
        pid_to_process_map = {p.pid: p for p in self.processes}
        sorted_pids = sorted(pid_to_process_map.keys())

        for pid in sorted_pids:
            process = pid_to_process_map[pid]
            priority_display = process.priority if process.priority is not None else 'N/A'
            print(f"  进程 {process.pid} (到达:{process.arrival_time}, 总运行:{process.run_time}, 优先级:{priority_display})")

        print("\n执行序列:")
        for ex in self.execution_sequence:
            print(f"  时间 [{ex['start_time']}-{ex['end_time']}]: 进程 {ex['pid']} ({ex['status']})")
        
        print("\n进程性能指标:")
        headers = ["进程ID", "到达", "总运行", "完成", "周转", "等待", "响应"]
        print("  " + " | ".join(h.ljust(6) for h in headers))
        print("  " + "-" * (sum(len(h.ljust(6)) for h in headers) + len(headers) * 3 -1 ))
        
        for pid in sorted_pids:
            process = pid_to_process_map[pid]
            if process.completion_time is not None:
                row = [
                    str(process.pid).ljust(6), str(process.arrival_time).ljust(6),
                    str(process.run_time).ljust(6), str(process.completion_time).ljust(6),
                    f"{process.turnaround_time:.2f}".ljust(6) if process.turnaround_time is not None else "N/A".ljust(6),
                    f"{process.waiting_time:.2f}".ljust(6) if process.waiting_time is not None else "N/A".ljust(6),
                    f"{process.response_time:.2f}".ljust(6) if process.response_time is not None else "N/A".ljust(6)
                ]
                print("  " + " | ".join(row))
            else: # Process might not have completed if simulation ended early or error
                print(f"  进程 {pid} 未完成.")

        metrics = self.calculate_system_metrics()
        print("\n系统性能指标:")
        print(f"  CPU 利用率: {metrics['cpu_utilization']*100:.2f}%")
        print(f"  吞吐量: {metrics['throughput']:.2f} 进程/时间单位")
        print(f"  平均周转时间: {metrics['avg_turnaround_time']:.2f} 时间单位")
        print(f"  平均等待时间: {metrics['avg_waiting_time']:.2f} 时间单位")
        print(f"  平均响应时间: {metrics['avg_response_time']:.2f} 时间单位")

    def visualize(self) -> None:
        process_colors = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Ensure all PIDs from execution_sequence have a color, including "空闲"
        all_pids_in_sequence = sorted(list(set(ex['pid'] for ex in self.execution_sequence if ex['pid'] != "空闲")))
        
        idx = 0
        for pid in all_pids_in_sequence:
            process_colors[pid] = colors[idx % len(colors)]
            idx +=1
        process_colors["空闲"] = '#D3D3D3' # Light grey for idle

        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.7, 0.3],
            specs=[[{"type": "scatter"}], [{"type": "table"}]],
            subplot_titles=(f"{self.name} 调度甘特图", "系统性能指标")
        )
        
        # Y-axis labels for Gantt chart (processes + Idle)
        y_labels = [f"进程 {pid}" for pid in all_pids_in_sequence]
        if any(ex['pid'] == "空闲" for ex in self.execution_sequence):
             y_labels.append("空闲")
        y_labels.reverse() # To have P1 at top or as desired

        for ex in self.execution_sequence:
            if ex["status"] != "抢占":
                y_val = f"进程 {ex['pid']}" if ex['pid'] != "空闲" else "空闲"
                fig.add_trace(
                    go.Bar(
                        x=[ex["end_time"] - ex["start_time"]], y=[y_val],
                        orientation='h', base=ex["start_time"],
                        marker=dict(color=process_colors.get(ex["pid"], '#000000')), # Default to black if PID not found
                        text=f"{ex['pid']}: {ex['start_time']}-{ex['end_time']}",
                        hoverinfo="text", showlegend=False
                    ), row=1, col=1
                )
        
        metrics = self.calculate_system_metrics()
        fig.add_trace(
            go.Table(
                header=dict(values=['指标', '值'], font=dict(size=14), align='center'),
                cells=dict(
                    values=[
                        ['CPU 利用率', '吞吐量', '平均周转时间', '平均等待时间', '平均响应时间'],
                        [f"{metrics['cpu_utilization']*100:.2f}%", f"{metrics['throughput']:.2f} 进程/时间单位",
                         f"{metrics['avg_turnaround_time']:.2f} 时间单位", f"{metrics['avg_waiting_time']:.2f} 时间单位",
                         f"{metrics['avg_response_time']:.2f} 时间单位"]
                    ], font=dict(size=12), align='center'
                )
            ), row=2, col=1
        )
        
        fig.update_layout(
            title_text=f"{self.name} 调度算法可视化", title_font_size=20,
            barmode='stack',
            yaxis_title='进程', yaxis_autorange="reversed", # yaxis_categoryorder='array', yaxis_categoryarray=y_labels,
            xaxis_title='时间', xaxis_dtick=1, xaxis_showgrid=True,
            height=800, font_family="SimHei, Arial" # Added Arial as fallback
        )
        fig.show()


class FCFSScheduler(Scheduler):
    def __init__(self, processes: List[Process], priority_logic_higher_is_better: bool):
        super().__init__(processes, "先来先服务(FCFS)", priority_logic_higher_is_better)

    def select_process(self) -> Optional[Process]:
        if not self.ready_queue: return None
        return sorted(self.ready_queue, key=lambda p: (p.arrival_time, p.pid))[0]

class NonPreemptiveSJFScheduler(Scheduler):
    def __init__(self, processes: List[Process], priority_logic_higher_is_better: bool):
        super().__init__(processes, "非抢占式短作业优先(SJF)", priority_logic_higher_is_better)

    def select_process(self) -> Optional[Process]:
        if self.current_process and self.current_process.remaining_time > 0: # Non-preemptive
            return self.current_process
        if not self.ready_queue: return None
        return sorted(self.ready_queue, key=lambda p: (p.run_time, p.arrival_time, p.pid))[0]

class PreemptiveSJFScheduler(Scheduler): # SRTF
    def __init__(self, processes: List[Process], priority_logic_higher_is_better: bool):
        super().__init__(processes, "抢占式短作业优先(SRTF)", priority_logic_higher_is_better)

    def is_preemptive(self) -> bool: return True

    def select_process(self) -> Optional[Process]:
        candidates = self.ready_queue.copy()
        if self.current_process and self.current_process.remaining_time > 0:
            if self.current_process not in candidates: # Ensure current is considered if it was removed by mistake
                 candidates.append(self.current_process)
        if not candidates: return None
        return sorted(candidates, key=lambda p: (p.remaining_time, p.arrival_time, p.pid))[0]

    def get_execution_time(self, process: Process) -> int:
        # Run until current process finishes, or a new process arrives that might be shorter
        time_to_finish = process.remaining_time
        
        time_to_next_arrival = float('inf')
        for p_other in self.processes:
            if p_other.arrival_time > self.current_time and p_other not in self.completed_processes and p_other != process :
                time_to_next_arrival = min(time_to_next_arrival, p_other.arrival_time - self.current_time)
        
        return min(time_to_finish, time_to_next_arrival, 1) # Execute at least 1 time unit, or until next event. Max 1 for fine-grained preemption.


class NonPreemptivePriorityScheduler(Scheduler):
    def __init__(self, processes: List[Process], priority_logic_higher_is_better: bool):
        super().__init__(processes, "非抢占式优先级调度", priority_logic_higher_is_better)
        self._check_priorities()
        self.config["优先级逻辑"] = "数值越大优先级越高" if self.priority_logic_higher_is_better else "数值越小优先级越高"

    def _check_priorities(self):
        if any(p.priority is None for p in self.processes):
            raise ValueError("所有进程必须定义优先级才能使用此调度算法。")

    def select_process(self) -> Optional[Process]:
        if self.current_process and self.current_process.remaining_time > 0:
            return self.current_process
        if not self.ready_queue: return None
        return sorted(self.ready_queue, key=self._get_sort_key_for_priority)[0]


class PreemptivePriorityScheduler(Scheduler):
    def __init__(self, processes: List[Process], priority_logic_higher_is_better: bool):
        super().__init__(processes, "抢占式优先级调度", priority_logic_higher_is_better)
        self._check_priorities()
        self.config["优先级逻辑"] = "数值越大优先级越高" if self.priority_logic_higher_is_better else "数值越小优先级越高"

    def _check_priorities(self):
        if any(p.priority is None for p in self.processes):
            raise ValueError("所有进程必须定义优先级才能使用此调度算法。")

    def is_preemptive(self) -> bool: return True

    def select_process(self) -> Optional[Process]:
        candidates = self.ready_queue.copy()
        if self.current_process and self.current_process.remaining_time > 0:
             if self.current_process not in candidates:
                 candidates.append(self.current_process)
        if not candidates: return None
        return sorted(candidates, key=self._get_sort_key_for_priority)[0]

    def get_execution_time(self, process: Process) -> int:
        # Similar to SRTF, run for 1 unit or until next event.
        time_to_finish = process.remaining_time
        time_to_next_arrival_of_potentially_higher_priority = float('inf')

        for p_other in self.processes:
            if p_other.arrival_time > self.current_time and \
               p_other not in self.completed_processes and \
               p_other != process and p_other.priority is not None:
                # Check if p_other could preempt current process
                # This needs careful comparison based on priority logic
                p_other_eff_prio = -p_other.priority if self.priority_logic_higher_is_better else p_other.priority
                process_eff_prio = -process.priority if self.priority_logic_higher_is_better else process.priority
                if p_other_eff_prio < process_eff_prio : # p_other has higher priority
                    time_to_next_arrival_of_potentially_higher_priority = min(
                        time_to_next_arrival_of_potentially_higher_priority, 
                        p_other.arrival_time - self.current_time
                    )
        
        return min(time_to_finish, time_to_next_arrival_of_potentially_higher_priority, 1) # Max 1 unit for preemption check


class RoundRobinScheduler(Scheduler):
    def __init__(self, processes: List[Process], time_quantum: int, priority_logic_higher_is_better: bool):
        super().__init__(processes, "时间片轮转(RR)", priority_logic_higher_is_better)
        self.time_quantum = time_quantum
        self.config["时间片大小"] = time_quantum

    def is_preemptive(self) -> bool: return True

    def select_process(self) -> Optional[Process]:
        if not self.ready_queue: return None
        # RR typically picks from the head of a FIFO queue.
        # Sort by arrival time to break ties if multiple added in one update_ready_queue call.
        return sorted(self.ready_queue, key=lambda p: p.arrival_time)[0]


    def get_execution_time(self, process: Process) -> int:
        return min(self.time_quantum, process.remaining_time)

    def handle_uncompleted_process(self, process: Process, executed_duration: int) -> None:
        if process.remaining_time > 0:
            if process not in self.ready_queue: # Add to end of ready queue if not there
                self.ready_queue.append(process)
            # If it was already in ready_queue (e.g. preemption by higher prio scheduler), ensure it's moved to end.
            elif process in self.ready_queue:
                 self.ready_queue.remove(process)
                 self.ready_queue.append(process)


class MultiLevelQueueScheduler(Scheduler):
    def __init__(self, processes: List[Process], queue_configs: List[Dict], priority_logic_higher_is_better: bool):
        super().__init__(processes, "多级队列调度", priority_logic_higher_is_better)
        self.queue_configs = queue_configs
        self.queues: List[List[Process]] = [[] for _ in range(len(queue_configs))]
        
        # Assign processes to queues based on their priority (smaller means higher queue level)
        # This interpretation of process.priority is specific to MLQS initial assignment.
        for process in self.processes:
            if process.priority is None:
                raise ValueError(f"进程 {process.pid} 必须有优先级才能用于多级队列调度（用于初始队列分配）。")
            # Process priority here maps to queue index, 0 being highest.
            queue_idx = min(process.priority, len(self.queues) - 1)
            process.current_queue = queue_idx
        
        config_str = []
        for i, cfg in enumerate(queue_configs):
            q_alg = cfg["algorithm"]
            q_detail = f"队列{i}:{q_alg}"
            if q_alg == "RR": q_detail += f"(TQ:{cfg['time_quantum']})"
            config_str.append(q_detail)
        self.config["队列配置"] = ", ".join(config_str)
        if any(qc["algorithm"] == "Priority" for qc in queue_configs):
             self.config["优先级逻辑"] = "数值越大优先级越高" if self.priority_logic_higher_is_better else "数值越小优先级越高"


    def update_ready_queue(self) -> None: # Override for multiple queues
        for process in self.processes:
            if (process.arrival_time <= self.current_time and 
                process.remaining_time > 0 and 
                process not in self.completed_processes and
                process != self.current_process):
                # Check if already in any queue
                in_any_queue = any(process in q for q in self.queues)
                if not in_any_queue:
                    self.queues[process.current_queue].append(process)
        
        # Base ready_queue is sum of all queues (for compatibility/checks, not primary use)
        self.ready_queue = [p for q in self.queues for p in q]


    def select_process(self) -> Optional[Process]:
        for i, queue in enumerate(self.queues):
            if not queue: continue
            
            alg_config = self.queue_configs[i]
            algorithm = alg_config["algorithm"]

            if algorithm == "FCFS":
                return sorted(queue, key=lambda p: (p.arrival_time, p.pid))[0]
            elif algorithm == "SJF": # Non-preemptive SJF for queue
                return sorted(queue, key=lambda p: (p.run_time, p.arrival_time, p.pid))[0]
            elif algorithm == "Priority":
                return sorted(queue, key=self._get_sort_key_for_priority)[0]
            elif algorithm == "RR":
                return sorted(queue, key=lambda p: p.arrival_time)[0] # Pick earliest arrived for RR queue head
        return None

    def is_preemptive(self) -> bool: # MLQ is preemptive between queues. Within a queue, depends on its algo.
        # If a higher priority queue becomes non-empty, it preempts.
        if self.current_process:
            current_q_idx = self.current_process.current_queue
            for i in range(current_q_idx): # Check higher priority queues
                if self.queues[i]:
                    return True # Higher priority queue has a process, preempt!
            
            # If no higher priority queue preemption, check current queue's algorithm
            # This part is tricky; preemption here means if the current process's own algorithm is preemptive
            # The main loop's preemption check will handle SRTF/Prio-P like preemption within a queue correctly
            # if select_process returns a different process.
            # For RR, get_execution_time limits it.
            q_algo = self.queue_configs[current_q_idx]["algorithm"]
            if q_algo == "RR": return True 
            # If SJF/Priority selected for queue, they are effectively SRTF/Prio-P due to main loop logic
            # if a new better process arrives in *this* queue.
        return False # Default to non-preemptive if no specific condition met


    def get_execution_time(self, process: Process) -> int:
        q_idx = process.current_queue
        alg_config = self.queue_configs[q_idx]
        algorithm = alg_config["algorithm"]

        # Preemption by higher queue: run until a process arrives in a higher queue
        time_to_higher_q_arrival = float('inf')
        for check_q_idx in range(q_idx): # Iterate higher priority queues
            for p_other in self.processes: # Check all processes
                if p_other.current_queue == check_q_idx and \
                   p_other.arrival_time > self.current_time and \
                   p_other not in self.completed_processes:
                    time_to_higher_q_arrival = min(time_to_higher_q_arrival, p_other.arrival_time - self.current_time)
        
        run_limit_by_algo = process.remaining_time
        if algorithm == "RR":
            run_limit_by_algo = min(alg_config["time_quantum"], process.remaining_time)
        
        # For SJF/Priority in a queue, they run to completion unless preempted by higher queue or new arrival in *same* queue
        # The '1' for SRTF/PrioP in main schedulers is for fine-grained check. Here, higher Q preemption is main.
        
        return min(run_limit_by_algo, time_to_higher_q_arrival, 1 if algorithm in ["SJF", "Priority"] else process.remaining_time)


    def handle_uncompleted_process(self, process: Process, executed_duration: int) -> None:
        if process.remaining_time > 0:
            q_idx = process.current_queue
            # Process stays in its current queue in MLQS (no demotion/promotion)
            # Add to end if RR, otherwise just add back for re-sorting.
            # Current select_process for RR in MLQS already picks earliest; simple append is fine.
            if process not in self.queues[q_idx]:
                 self.queues[q_idx].append(process)
            elif self.queue_configs[q_idx]["algorithm"] == "RR": # Ensure it goes to end for RR
                 self.queues[q_idx].remove(process)
                 self.queues[q_idx].append(process)



class MultilevelFeedbackQueueScheduler(Scheduler):
    def __init__(self, processes: List[Process], queue_count: int, time_quantums: List[int], 
                 priority_logic_higher_is_better: bool, enable_aging: bool = False, aging_threshold: int = 10):
        super().__init__(processes, "多级反馈队列调度", priority_logic_higher_is_better)
        self.queue_count = queue_count
        self.time_quantums = time_quantums # TQ for Q0 to Qn-2. Qn-1 is FCFS.
        self.queues: List[List[Process]] = [[] for _ in range(queue_count)]
        self.enable_aging = enable_aging
        self.aging_threshold = aging_threshold
        # Store original PIDs for aging map as Process objects might be new copies per run
        self.process_wait_since_last_run_or_demotion = {p.pid: 0 for p in self.processes}

        for process in self.processes:
            process.current_queue = 0 # All start in highest queue

        self.config["队列数量"] = queue_count
        self.config["各队列时间片"] = ", ".join(map(str, time_quantums[:-1])) + f", 最后一个队列: FCFS"
        self.config["启用老化"] = "是" if enable_aging else "否"
        if enable_aging: self.config["老化阈值"] = aging_threshold

    def update_ready_queue(self) -> None:
        for process in self.processes:
            if (process.arrival_time <= self.current_time and 
                process.remaining_time > 0 and 
                process not in self.completed_processes and
                process != self.current_process):
                
                in_any_queue = any(process in q for q in self.queues)
                if not in_any_queue:
                    # New or returning process, ensure it's in its current_queue
                    # (current_queue might have been changed by demotion/promotion)
                    self.queues[process.current_queue].append(process)
        
        if self.enable_aging:
            self._apply_aging()
        
        self.ready_queue = [p for q in self.queues for p in q]

    def _apply_aging(self) -> None:
        # Increment wait time for all processes in queues (except Q0) that are not currently running
        for q_idx in range(1, self.queue_count): # Aging applies to Q1 onwards
            for process in self.queues[q_idx][:]: # Iterate copy for safe removal
                if process != self.current_process:
                    self.process_wait_since_last_run_or_demotion[process.pid] += 1 # Crude: increments each scheduling cycle
                
                if self.process_wait_since_last_run_or_demotion[process.pid] >= self.aging_threshold:
                    self.queues[q_idx].remove(process)
                    promoted_q_idx = max(0, q_idx - 1)
                    process.current_queue = promoted_q_idx
                    self.queues[promoted_q_idx].append(process)
                    self.process_wait_since_last_run_or_demotion[process.pid] = 0 # Reset wait time

    def select_process(self) -> Optional[Process]:
        for i, queue in enumerate(self.queues):
            if not queue: continue
            if i == self.queue_count - 1: # Last queue is FCFS
                return sorted(queue, key=lambda p: (p.arrival_time, p.pid))[0]
            else: # Higher queues are RR (effectively, pick first)
                return sorted(queue, key=lambda p: p.arrival_time)[0] # FIFO within RR queue
        return None

    def is_preemptive(self) -> bool: return True

    def get_execution_time(self, process: Process) -> int:
        q_idx = process.current_queue
        if q_idx == self.queue_count - 1: # FCFS queue
            return process.remaining_time
        else: # RR queue
            return min(self.time_quantums[q_idx], process.remaining_time)

    def handle_uncompleted_process(self, process: Process, executed_duration: int) -> None:
        if process.remaining_time > 0:
            current_q_idx = process.current_queue
            self.process_wait_since_last_run_or_demotion[process.pid] = 0 # Ran, so reset its wait time

            demote = False
            if current_q_idx < self.queue_count - 1: # Not in FCFS queue
                # If it used its full allocated time slice for this queue
                if executed_duration >= self.time_quantums[current_q_idx]:
                    demote = True
            
            if demote:
                next_q_idx = min(current_q_idx + 1, self.queue_count - 1)
                process.current_queue = next_q_idx
                self.queues[next_q_idx].append(process)
            else: # Stays in current queue (e.g., preempted early or TQ not fully used)
                self.queues[current_q_idx].append(process) # Add to end for RR behavior


# --- Helper Functions ---
def clear_screen():
    os.system('cls' if platform.system() == "Windows" else 'clear')

def validate_processes_for_priority(processes: List[Process], algorithm_name: str) -> bool:
    needs_priority_algos = [
        "非抢占式优先级调度", "抢占式优先级调度", 
        "多级队列调度" # Because it uses priority for initial queue assignment
    ]
    if algorithm_name in needs_priority_algos:
        if any(p.priority is None for p in processes):
            print(f"错误: 算法 '{algorithm_name}' 要求所有进程都定义优先级。")
            return False
    return True

def get_integer_input(prompt: str, min_val: int = 0, max_val: int = float('inf')) -> int:
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val: return val
            print(f"请输入 {min_val} 到 {max_val} 之间的整数。")
        except ValueError: print("无效输入，请输入整数。")

def get_yes_no_input(prompt: str) -> bool:
    while True:
        res = input(prompt + " (y/n): ").strip().lower()
        if res in ['y', 'yes', '是', '1']: return True
        if res in ['n', 'no', '否', '0']: return False
        print("无效输入，请输入 y/n 或 是/否。")

def manual_input_processes(priority_logic_higher_is_better: bool) -> List[Process]:
    processes = []
    count = get_integer_input("请输入进程数量: ", 1, 20)
    
    # Ask once if priority is generally needed for any algorithm they might pick
    any_algo_needs_priority = get_yes_no_input("是否计划使用需要优先级的调度算法 (如优先级调度, 多级队列)?")

    for i in range(count):
        print(f"\n--- 输入进程 {i+1} ---")
        pid = i + 1
        at = get_integer_input(f"  到达时间: ", 0)
        bt = get_integer_input(f"  运行时间: ", 1)
        prio = None
        if any_algo_needs_priority:
            prio_prompt_detail = "(数值越小优先级越高)"
            if priority_logic_higher_is_better:
                prio_prompt_detail = "(数值越大优先级越高)"
            
            prio_info = (f"  优先级 {prio_prompt_detail}.\n"
                         f"    (注意: 对于多级队列调度, 此值用作初始队列索引, 0=最高级队列): ")
            prio = get_integer_input(prio_info, 0)
        processes.append(Process(pid, at, bt, prio))
    return processes

def random_generate_processes(priority_logic_higher_is_better: bool) -> List[Process]:
    processes = []
    count = get_integer_input("生成进程数量: ", 1, 20)
    max_at = get_integer_input("最大到达时间: ", 0)
    max_bt = get_integer_input("最大运行时间: ", 1)
    
    any_algo_needs_priority = get_yes_no_input("是否为进程生成优先级?")
    
    for i in range(count):
        pid = i + 1
        at = random.randint(0, max_at)
        bt = random.randint(1, max_bt)
        prio = None
        if any_algo_needs_priority:
            prio = random.randint(0, 9) # Example range for priority
        processes.append(Process(pid, at, bt, prio))
    print("\n随机生成的进程:")
    for p in processes: print(f"  {p}")
    input("按回车键继续...")
    return processes

def run_scheduling_simulation():
    global PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER # Allow modification

    clear_screen()
    print("====== 操作系统调度算法可视化工具 ======\n")

    print("请选择优先级判定逻辑:")
    print("1. 数值越小，优先级越高 (例如: 0 比 1 优先级高)")
    print("2. 数值越大，优先级越高 (例如: 1 比 0 优先级高)")
    choice = get_integer_input("输入选择 (1/2): ", 1, 2)
    PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER = (choice == 2)

    print("\n选择进程数据输入方式:")
    print("1. 手动输入")
    print("2. 随机生成")
    input_mode = get_integer_input("输入选择 (1/2): ", 1, 2)

    original_processes: List[Process]
    if input_mode == 1:
        original_processes = manual_input_processes(PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER)
    else:
        original_processes = random_generate_processes(PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER)

    while True: # Loop for running multiple algorithms on the same process set
        # Create fresh Process objects for each simulation run from originals
        current_processes_to_run = [
            Process(p.pid, p.arrival_time, p.run_time, p.priority) 
            for p in original_processes
        ]

        clear_screen()
        print("\n当前进程集:")
        for p in current_processes_to_run: print(f"  {p}")
        
        print("\n请选择调度算法:")
        print("1. FCFS")
        print("2. 非抢占SJF")
        print("3. 抢占SJF (SRTF)")
        print("4. 非抢占优先级")
        print("5. 抢占优先级")
        print("6. 时间片轮转 (RR)")
        print("7. 多级队列")
        print("8. 多级反馈队列")
        algo_choice = get_integer_input("输入选择 (1-8): ", 1, 8)

        scheduler: Optional[Scheduler] = None
        algo_name_map = {
            1: "先来先服务(FCFS)", 2: "非抢占式短作业优先(SJF)", 3: "抢占式短作业优先(SRTF)",
            4: "非抢占式优先级调度", 5: "抢占式优先级调度", 6: "时间片轮转(RR)",
            7: "多级队列调度", 8: "多级反馈队列调度"
        }
        selected_algo_name = algo_name_map.get(algo_choice, "未知算法")

        if not validate_processes_for_priority(current_processes_to_run, selected_algo_name):
            if not get_yes_no_input("进程数据不满足算法要求。是否重新选择算法或退出? (y=重选, n=退出程序)"):
                return
            continue # Restart algorithm selection loop


        if algo_choice == 1:
            scheduler = FCFSScheduler(current_processes_to_run, PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER)
        elif algo_choice == 2:
            scheduler = NonPreemptiveSJFScheduler(current_processes_to_run, PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER)
        elif algo_choice == 3:
            scheduler = PreemptiveSJFScheduler(current_processes_to_run, PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER)
        elif algo_choice == 4:
            scheduler = NonPreemptivePriorityScheduler(current_processes_to_run, PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER)
        elif algo_choice == 5:
            scheduler = PreemptivePriorityScheduler(current_processes_to_run, PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER)
        elif algo_choice == 6:
            tq = get_integer_input("输入时间片大小: ", 1)
            scheduler = RoundRobinScheduler(current_processes_to_run, tq, PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER)
        elif algo_choice == 7:
            num_q = get_integer_input("输入多级队列数量 (2-5): ", 2, 5)
            q_configs = []
            print("\n--- 配置各级队列 ---")
            for i in range(num_q):
                print(f"\n配置队列 {i} (高优先级队列):")
                print("  1. FCFS  2. SJF (非抢占)  3. Priority  4. RR")
                q_algo_idx = get_integer_input(f"  队列 {i} 算法 (1-4): ", 1, 4)
                cfg = {}
                if q_algo_idx == 1: cfg["algorithm"] = "FCFS"
                elif q_algo_idx == 2: cfg["algorithm"] = "SJF"
                elif q_algo_idx == 3: cfg["algorithm"] = "Priority"
                elif q_algo_idx == 4:
                    cfg["algorithm"] = "RR"
                    cfg["time_quantum"] = get_integer_input(f"  队列 {i} RR时间片: ", 1)
                q_configs.append(cfg)
            scheduler = MultiLevelQueueScheduler(current_processes_to_run, q_configs, PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER)
        elif algo_choice == 8:
            num_q = get_integer_input("输入多级反馈队列数量 (2-5): ", 2, 5)
            tqs = []
            print("\n--- 配置各队列时间片 (最后一个队列为FCFS) ---")
            for i in range(num_q - 1):
                tqs.append(get_integer_input(f"  队列 {i} 时间片: ", 1))
            tqs.append(0) # Placeholder for FCFS queue (unused TQ)
            
            use_aging = get_yes_no_input("是否启用老化机制?")
            aging_t = 10
            if use_aging:
                aging_t = get_integer_input("输入老化阈值 (等待多少调度周期后提升): ", 1)
            scheduler = MultilevelFeedbackQueueScheduler(current_processes_to_run, num_q, tqs, PRIORITY_LOGIC_HIGHER_VALUE_IS_BETTER, use_aging, aging_t)

        if scheduler:
            try:
                scheduler.run()
                scheduler.print_results()
                if get_yes_no_input("\n是否显示可视化甘特图?"):
                    scheduler.visualize()
            except Exception as e:
                print(f"\n!!! 调度过程中发生错误: {e} !!!")
                import traceback
                traceback.print_exc()


        if not get_yes_no_input("\n是否使用当前进程集运行其他调度算法?"):
            break # Exit simulation loop

    print("\n模拟结束。感谢使用！")

if __name__ == "__main__":
    # Ensure Plotly can find fonts if running in a restricted environment
    # This is generally not needed for standard desktop execution.
    # If issues with Chinese fonts in Plotly, ensure SimHei or a similar font is installed.
    run_scheduling_simulation()