import heapq
import random
import math
from typing import List, Dict, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd # Explicitly import pandas
import plotly.express as px # Explicitly import plotly.express

# --- Add warning filter ---
import warnings
# To suppress the specific FutureWarning from Plotly Express interacting with Pandas
# This is a workaround for a warning originating from the library,
# not a fix for the library itself. Future versions of Plotly/Pandas may resolve this.
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly.express._core")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.groupby.generic") # Also from pandas itself

# --- Process Class ---
class Process:
    def __init__(self, pid: str, arrival_time: int, run_time: int, priority: Optional[int] = None):
        self.pid: str = pid
        self.arrival_time: int = arrival_time
        self.run_time: int = run_time
        # Lower numerical value means higher priority for scheduler logic
        # Internally, if None, it's treated as lowest priority
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
        self.last_run_time: int = 0 # for aging in MLFQ
        self.assigned_queue_index: int = 0 # For MLQ

    def __lt__(self, other: 'Process') -> bool:
        # Default comparison for heapq, typically based on arrival_time for FCFS-like behavior
        # Can be overridden by how elements are pushed into a heapq if tuples are used.
        return self.arrival_time < other.arrival_time

    def __repr__(self) -> str:
        priority_display = self.priority if self.priority is not None else 'N/A'
        return (f"Process(pid={self.pid}, arrival={self.arrival_time}, run={self.run_time}, "
                f"priority={priority_display}, remaining={self.remaining_time})")

    def get_priority_val(self) -> float:
        """Returns the priority value, defaulting to infinity for undefined priorities (lowest)."""
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
        # self.assigned_queue_index remains as it's part of initial config for MLQ

# --- Scheduler Base Class ---
class Scheduler:
    def __init__(self, processes: List[Process]):
        # Sort original processes by arrival time primarily for consistent initial handling
        self.processes_master_list: List[Process] = sorted([p for p in processes], key=lambda p: (p.arrival_time, p.pid))
        for p in self.processes_master_list:
            p.reset() # Ensure processes are in a clean state for the simulation

        self.completed_processes: List[Process] = []
        self.gantt_chart_data: List[Dict[str, Any]] = []
        self.current_time: int = 0
        self.total_cpu_busy_time: int = 0

        # ready_queue type can vary (list, heapq, etc.) - managed by subclasses
        self.ready_queue: Any = []

        self.algorithm_name: str = "Scheduler Base"
        self.algorithm_params: Dict[str, Any] = {}

    def _update_processes_to_ready_state(self, pending_processes_list: List[Process]) -> List[Process]:
        """
        Moves processes from the pending_processes_list to the scheduler's ready queue
        if they have arrived by the self.current_time.
        Returns the list of processes added to the ready queue in this step.
        Specific enqueue logic is handled by subclasses overriding _enqueue_process.
        """
        newly_ready_processes = []
        # Iterate over a copy of pending_processes_list for safe removal
        for process in list(pending_processes_list):
            if process.arrival_time <= self.current_time:
                self._enqueue_process(process)
                pending_processes_list.remove(process)
                newly_ready_processes.append(process)
        return newly_ready_processes

    def _enqueue_process(self, process: Process) -> None:
        """
        Adds a process to the ready queue.
        This method should be overridden by subclasses to implement
        specific ready queue structures (e.g., heapq for priority queues).
        Default is simple list append, suitable for basic RR or to be sorted later.
        """
        if process not in self.ready_queue: # Avoid duplicates if logic allows
             self.ready_queue.append(process)


    def _calculate_metrics(self, process: Process) -> None:
        if process.completion_time is not None and process.start_time is not None: # process.start_time should always be set if completion_time is
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.run_time
            # Response time is calculated when process first starts execution
            if process.response_time is None: # Should have been set at first run
                process.response_time = process.start_time - process.arrival_time
        else:
            # This case should ideally not happen for a completed process
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
        # Ensure response_time is not None before summing
        total_response_time = sum(p.response_time for p in self.completed_processes if p.response_time is not None)

        num_completed = len(self.completed_processes)

        max_completion_time = 0
        if self.completed_processes: # Ensure there are completed processes
             max_val = max(p.completion_time for p in self.completed_processes if p.completion_time is not None)
             if max_val is not None:
                 max_completion_time = max_val

        total_simulation_time = max(1, max_completion_time) # Avoid division by zero.

        cpu_utilization = (self.total_cpu_busy_time / total_simulation_time) * 100 if total_simulation_time > 0 else 0.0
        throughput = num_completed / total_simulation_time if total_simulation_time > 0 else 0.0

        return {
            "CPU Utilization": round(cpu_utilization, 2),
            "Throughput": round(throughput, 2),
            "Average Turnaround Time": round(total_turnaround_time / num_completed, 2) if num_completed > 0 else 0.0,
            "Average Waiting Time": round(total_waiting_time / num_completed, 2) if num_completed > 0 else 0.0,
            "Average Response Time": round(total_response_time / num_completed, 2) if num_completed > 0 else 0.0
        }

    def run(self) -> None:
        # This method must be implemented by subclasses
        raise NotImplementedError("The run method must be implemented by subclasses.")

    def _add_gantt_entry(self, process_pid: str, start: int, finish: int, status: str = "Executing"):
        # Always print the console log for the event
        print(f"时间 [{start}-{finish}]: 进程 [{process_pid}] ({status})")

        # Logic for adding to Gantt chart data (visual representation)
        if finish <= start:
            # For Gantt chart, zero or negative duration tasks are problematic.
            if status == "Executing":
                # An "Executing" status with zero duration means it was scheduled but didn't run effectively for a time block.
                print(f"  (Gantt: 执行事件 {process_pid} 时长为0或负, 不添加到甘特图)")
                return
            else:
                # For non-executing events like "抢占", "完成" that might be instantaneous.
                # These are primarily console logs; adding them as zero-width to Gantt is often messy.
                print(f"  (Gantt: 非执行事件 {process_pid} 时长为0或负, 不添加到甘特图)")
                return # Current choice: skip zero/negative duration non-executing events for cleaner Gantt

        self.gantt_chart_data.append({
            "Task": process_pid, # Y-axis: Process ID
            "Start": start,      # X-axis start
            "Finish": finish,    # X-axis end
            "Resource": process_pid, # Color by process ID
            "Status": status     # Additional info, not directly used by px.timeline color/y
        })


    def print_initial_processes(self):
        print("\n初始进程信息:")
        for p in self.processes_master_list: # Use the master list for initial state
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
        df = pd.DataFrame() # Initialize an empty DataFrame

        if gantt_possible_initially:
            print(f"\n原始甘特图数据点数量: {len(self.gantt_chart_data)}")
            df = pd.DataFrame(self.gantt_chart_data)
            # Ensure Start and Finish are numeric for plotting and filtering
            df['Start'] = pd.to_numeric(df['Start'], errors='coerce')
            df['Finish'] = pd.to_numeric(df['Finish'], errors='coerce')
            df.dropna(subset=['Start', 'Finish'], inplace=True) # Remove rows where conversion failed

            # Filter out any zero-duration or negative-duration segments
            original_rows = len(df)
            df = df[df['Finish'] > df['Start']]
            filtered_rows = len(df)
            if original_rows > 0 and filtered_rows == 0 :
                print("甘特图数据在过滤后为空 (所有有效执行片段时长为0或起点不小于终点)。")
            elif original_rows > filtered_rows:
                 print(f"甘特图数据从 {original_rows} 条过滤为 {filtered_rows} 条 (移除了无效时长的片段)。")


        gantt_display_possible = not df.empty

        if not gantt_display_possible:
            print("没有有效的甘特图数据可供显示。将仅显示性能指标表格。")
            if not gantt_possible_initially and not self.completed_processes:
                 print("并且似乎没有任何进程完成，可能模拟未正确执行或无进程输入。")
                 return # Nothing to show at all


        # Create metrics table data (always attempt this if completed_processes exist)
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

        if gantt_display_possible: # If there is valid Gantt data to display
            rows = 2
            row_heights = [0.7, 0.3] # Give more space to Gantt
            specs = [[{"type": "xy"}], [{"type": "table"}]] # Gantt on top, table below
            subplot_titles_list = (f"{self.algorithm_name} 调度甘特图", f"{self.algorithm_name} 系统性能指标")


        final_fig = make_subplots(
            rows=rows, cols=1,
            row_heights=row_heights,
            specs=specs,
            vertical_spacing=0.15 if rows > 1 else 0.05,
            subplot_titles=subplot_titles_list # type: ignore
        )

        gantt_chart_row_index = 1
        metrics_table_row_index = 1 # Default if only table is shown

        if gantt_display_possible:
            # Create Gantt chart trace from the filtered DataFrame
            fig_gantt_internal = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource")
            for trace in fig_gantt_internal.data:
                final_fig.add_trace(trace, row=gantt_chart_row_index, col=1)

            max_finish_time = df['Finish'].max()
            min_start_time = df['Start'].min()
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
            metrics_table_row_index = 2 # Metrics table moves to the second row

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
            height=250 + (350 if gantt_display_possible else 0), # Adjust height
            title_text=f"{self.algorithm_name} 调度分析",
            title_x=0.5,
            font_family="SimHei, Noto Sans CJK SC, Microsoft YaHei, sans-serif",
            showlegend=False # Usually redundant if Y-axis and color are the same task
        )

        final_fig.show()

# --- FCFS Scheduler ---
class FCFS(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes)
        self.algorithm_name = "FCFS (先来先服务)"
        # FCFS ready queue is a simple list, processes added then sorted by arrival, then PID
        self.ready_queue: List[Process] = []

    def _enqueue_process(self, process: Process) -> None:
        if process not in self.ready_queue:
            self.ready_queue.append(process)
            # Keep ready_queue sorted for FCFS
            self.ready_queue.sort(key=lambda p: (p.arrival_time, p.pid))

    def run(self) -> None:
        pending_processes = list(self.processes_master_list) # Operate on a copy

        if not pending_processes: return

        # Initialize current_time to the arrival time of the first process if system is idle until then
        self.current_time = min(p.arrival_time for p in pending_processes) if pending_processes else 0

        num_total_processes = len(pending_processes)

        while len(self.completed_processes) < num_total_processes:
            self._update_processes_to_ready_state(pending_processes) # Enqueue logic handles sorting for FCFS

            if not self.ready_queue:
                if pending_processes: # If CPU is idle but future processes exist
                    self.current_time = min(p.arrival_time for p in pending_processes)
                    # After time jump, re-check for newly arrived processes
                    self._update_processes_to_ready_state(pending_processes)
                else: # No processes in ready queue and no pending processes
                    break

            if not self.ready_queue: # Still no ready processes (should not happen if pending_processes existed)
                break


            current_process = self.ready_queue.pop(0) # Get the earliest arrived process

            # If current_time is less than arrival_time (after a jump for an empty ready queue)
            # This means the CPU was idle until current_process.arrival_time
            if self.current_time < current_process.arrival_time:
                self.current_time = current_process.arrival_time

            if current_process.start_time is None:
                current_process.start_time = self.current_time
                current_process.response_time = current_process.start_time - current_process.arrival_time

            execution_start_time = self.current_time
            # Process runs for its full run_time
            finish_time = self.current_time + current_process.run_time
            self.total_cpu_busy_time += current_process.run_time
            self.current_time = finish_time # Update current time

            current_process.completion_time = finish_time
            current_process.remaining_time = 0 # Mark as fully executed

            self._calculate_metrics(current_process)
            self.completed_processes.append(current_process)
            self._add_gantt_entry(current_process.pid, execution_start_time, finish_time, "完成")

        self.completed_processes.sort(key=lambda p: (p.completion_time if p.completion_time is not None else float('inf'), p.pid))


# --- Non-Preemptive SJF Scheduler ---
class NonPreemptiveSJF(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes)
        self.algorithm_name = "非抢占式SJF (短作业优先)"
        self.ready_queue: List[Process] = []

    def _enqueue_process(self, process: Process) -> None:
        if process not in self.ready_queue:
            self.ready_queue.append(process)
            # Sort by run_time, then arrival_time, then pid
            self.ready_queue.sort(key=lambda p: (p.run_time, p.arrival_time, p.pid))

    def run(self) -> None:
        pending_processes = list(self.processes_master_list)
        if not pending_processes: return

        self.current_time = min(p.arrival_time for p in pending_processes) if pending_processes else 0
        num_total_processes = len(pending_processes)

        while len(self.completed_processes) < num_total_processes:
            self._update_processes_to_ready_state(pending_processes) # Enqueue logic handles SJF sorting

            if not self.ready_queue:
                if pending_processes:
                    self.current_time = min(p.arrival_time for p in pending_processes)
                    self._update_processes_to_ready_state(pending_processes) # Re-check after time jump
                else:
                    break

            if not self.ready_queue: break

            current_process = self.ready_queue.pop(0) # Shortest job is at the front

            if self.current_time < current_process.arrival_time:
                self.current_time = current_process.arrival_time

            if current_process.start_time is None:
                current_process.start_time = self.current_time
                current_process.response_time = current_process.start_time - current_process.arrival_time

            execution_start_time = self.current_time
            finish_time = self.current_time + current_process.run_time
            self.total_cpu_busy_time += current_process.run_time
            self.current_time = finish_time

            current_process.completion_time = finish_time
            current_process.remaining_time = 0

            self._calculate_metrics(current_process)
            self.completed_processes.append(current_process)
            self._add_gantt_entry(current_process.pid, execution_start_time, finish_time, "完成")

        self.completed_processes.sort(key=lambda p: (p.completion_time if p.completion_time is not None else float('inf'), p.pid))

# --- Preemptive SJF (SRTF) Scheduler ---
class SRTF(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes)
        self.algorithm_name = "抢占式SJF (最短剩余时间优先, SRTF)"
        # Min-heap: (remaining_time, arrival_time, pid, process_obj)
        self.ready_queue: List[Tuple[int, int, str, Process]] = []

    def _enqueue_process(self, process: Process) -> None:
        # Check if already in heap (can be complex, simpler to rely on not re-adding from pending)
        # For SRTF, if a process is preempted, it's added back.
        # This enqueue is mainly for initially arriving processes.
        heapq.heappush(self.ready_queue, (process.remaining_time, process.arrival_time, process.pid, process))

    def _re_enqueue_process(self, process: Process) -> None:
        """Specifically for adding a preempted process back to the ready queue."""
        heapq.heappush(self.ready_queue, (process.remaining_time, process.arrival_time, process.pid, process))

    def _dequeue_process(self) -> Optional[Process]:
        if not self.ready_queue:
            return None
        return heapq.heappop(self.ready_queue)[3] # Get the process object

    def run(self) -> None:
        pending_processes = list(self.processes_master_list)
        if not pending_processes and not self.ready_queue : return

        current_process: Optional[Process] = None
        last_gantt_segment_start_time = -1 # Tracks start of current execution slice for Gantt

        # Start time can be 0 or the arrival of the first process
        self.current_time = 0
        if self.processes_master_list:
             self.current_time = min(p.arrival_time for p in self.processes_master_list) if self.processes_master_list else 0


        num_total_processes = len(self.processes_master_list)
        while len(self.completed_processes) < num_total_processes:

            self._update_processes_to_ready_state(pending_processes)

            # Preemption Check or if current process completed
            if current_process:
                # Check if current_process should be preempted by a newly arrived or now-ready shorter process
                if self.ready_queue and self.ready_queue[0][0] < current_process.remaining_time:
                    # Preempt current_process
                    if last_gantt_segment_start_time != -1: # Log its execution up to now
                         self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, self.current_time, "抢占")
                    self._re_enqueue_process(current_process)
                    current_process = None
                    last_gantt_segment_start_time = -1

            # If no process is running (either preempted or completed, or first iteration)
            if not current_process:
                potential_next_process = self._dequeue_process()
                if potential_next_process:
                    current_process = potential_next_process
                    last_gantt_segment_start_time = self.current_time # New segment starts now
                    if current_process.start_time is None: # First time ever execution for this process
                        current_process.start_time = self.current_time
                        current_process.response_time = current_process.start_time - current_process.arrival_time
                else: # No process available in ready queue
                    if pending_processes: # If future processes exist, advance time
                        self.current_time = min(p.arrival_time for p in pending_processes)
                        continue # Re-evaluate at new current_time (arrivals, etc.)
                    elif len(self.completed_processes) < num_total_processes: # Idle tick if stuck but not all done
                         self.current_time +=1
                         continue
                    else: # All processes done or no path forward
                        break

            # Execute current process for one time unit
            if current_process:
                current_process.remaining_time -= 1
                self.total_cpu_busy_time += 1

                # Check for completion AFTER executing for this tick
                if current_process.remaining_time == 0:
                    current_process.completion_time = self.current_time + 1 # Completes at END of this time unit
                    self._calculate_metrics(current_process)
                    self.completed_processes.append(current_process)
                    if last_gantt_segment_start_time != -1:
                         self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, current_process.completion_time, "完成")
                    current_process = None
                    last_gantt_segment_start_time = -1

            self.current_time += 1 # Advance global time

        self.completed_processes.sort(key=lambda p: (p.completion_time if p.completion_time is not None else float('inf'), p.pid))


# --- Non-Preemptive Priority Scheduler ---
class NonPreemptivePriority(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes)
        self.algorithm_name = "非抢占式优先级调度"
        self.ready_queue: List[Process] = [] # Will be sorted by priority

    def _enqueue_process(self, process: Process) -> None:
        if process not in self.ready_queue:
            self.ready_queue.append(process)
            # Sort by priority (lower num = higher prio), then arrival, then pid
            self.ready_queue.sort(key=lambda p: (p.get_priority_val(), p.arrival_time, p.pid))

    def run(self) -> None:
        pending_processes = list(self.processes_master_list)
        if not pending_processes: return

        self.current_time = min(p.arrival_time for p in pending_processes) if pending_processes else 0
        num_total_processes = len(pending_processes)

        while len(self.completed_processes) < num_total_processes:
            self._update_processes_to_ready_state(pending_processes) # Enqueue sorts by priority

            if not self.ready_queue:
                if pending_processes:
                    self.current_time = min(p.arrival_time for p in pending_processes)
                    self._update_processes_to_ready_state(pending_processes) # Re-check after time jump
                else:
                    break

            if not self.ready_queue: break

            current_process = self.ready_queue.pop(0) # Highest priority process

            if self.current_time < current_process.arrival_time:
                self.current_time = current_process.arrival_time

            if current_process.start_time is None:
                current_process.start_time = self.current_time
                current_process.response_time = current_process.start_time - current_process.arrival_time

            execution_start_time = self.current_time
            finish_time = self.current_time + current_process.run_time
            self.total_cpu_busy_time += current_process.run_time
            self.current_time = finish_time

            current_process.completion_time = finish_time
            current_process.remaining_time = 0

            self._calculate_metrics(current_process)
            self.completed_processes.append(current_process)
            self._add_gantt_entry(current_process.pid, execution_start_time, finish_time, "完成")

        self.completed_processes.sort(key=lambda p: (p.completion_time if p.completion_time is not None else float('inf'), p.pid))


# --- Preemptive Priority Scheduler ---
class PreemptivePriority(Scheduler):
    def __init__(self, processes: List[Process]):
        super().__init__(processes)
        self.algorithm_name = "抢占式优先级调度"
        # Min-heap: (priority_val, arrival_time, pid, process_obj)
        self.ready_queue: List[Tuple[float, int, str, Process]] = []

    def _enqueue_process(self, process: Process) -> None:
        heapq.heappush(self.ready_queue, (process.get_priority_val(), process.arrival_time, process.pid, process))

    def _re_enqueue_process(self, process: Process) -> None:
        heapq.heappush(self.ready_queue, (process.get_priority_val(), process.arrival_time, process.pid, process))

    def _dequeue_process(self) -> Optional[Process]:
        if not self.ready_queue:
            return None
        return heapq.heappop(self.ready_queue)[3]

    def run(self) -> None:
        pending_processes = list(self.processes_master_list)
        if not pending_processes and not self.ready_queue: return

        current_process: Optional[Process] = None
        last_gantt_segment_start_time = -1

        self.current_time = 0
        if self.processes_master_list:
            self.current_time = min(p.arrival_time for p in self.processes_master_list) if self.processes_master_list else 0

        num_total_processes = len(self.processes_master_list)
        while len(self.completed_processes) < num_total_processes:

            self._update_processes_to_ready_state(pending_processes)

            if current_process:
                if self.ready_queue and self.ready_queue[0][0] < current_process.get_priority_val():
                    if last_gantt_segment_start_time != -1:
                         self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, self.current_time, "抢占")
                    self._re_enqueue_process(current_process)
                    current_process = None
                    last_gantt_segment_start_time = -1

            if not current_process:
                potential_next_process = self._dequeue_process()
                if potential_next_process:
                    current_process = potential_next_process
                    last_gantt_segment_start_time = self.current_time
                    if current_process.start_time is None:
                        current_process.start_time = self.current_time
                        current_process.response_time = current_process.start_time - current_process.arrival_time
                else:
                    if pending_processes:
                        self.current_time = min(p.arrival_time for p in pending_processes)
                        continue
                    elif len(self.completed_processes) < num_total_processes:
                         self.current_time +=1
                         continue
                    else:
                        break

            if current_process:
                current_process.remaining_time -= 1
                self.total_cpu_busy_time += 1

                if current_process.remaining_time == 0:
                    current_process.completion_time = self.current_time + 1
                    self._calculate_metrics(current_process)
                    self.completed_processes.append(current_process)
                    if last_gantt_segment_start_time != -1:
                        self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, current_process.completion_time, "完成")
                    current_process = None
                    last_gantt_segment_start_time = -1

            self.current_time += 1
        self.completed_processes.sort(key=lambda p: (p.completion_time if p.completion_time is not None else float('inf'), p.pid))


# --- Round Robin Scheduler ---
class RoundRobin(Scheduler):
    def __init__(self, processes: List[Process], time_quantum: int):
        super().__init__(processes)
        self.time_quantum: int = time_quantum
        self.algorithm_name = "时间片轮转 (Round Robin, RR)"
        self.algorithm_params = {"时间片大小": self.time_quantum}
        self.ready_queue: List[Process] = [] # FIFO queue for RR

    def _enqueue_process(self, process: Process) -> None:
        # Add to the end of the list-based queue
        if process not in self.ready_queue: # Avoid adding if already there (e.g. if current_process is also in a newly_arrived list)
            self.ready_queue.append(process)

    def _re_enqueue_process(self, process: Process) -> None:
        """For adding a process back after its time slice or preemption."""
        self.ready_queue.append(process)

    def _dequeue_process(self) -> Optional[Process]:
        if not self.ready_queue:
            return None
        return self.ready_queue.pop(0) # Get from the front

    def run(self) -> None:
        pending_processes = list(self.processes_master_list)
        if not pending_processes and not self.ready_queue: return

        current_process: Optional[Process] = None
        current_slice_used = 0
        last_gantt_segment_start_time = -1

        self.current_time = 0
        if self.processes_master_list:
            self.current_time = min(p.arrival_time for p in self.processes_master_list) if self.processes_master_list else 0

        num_total_processes = len(self.processes_master_list)

        # Initial population of ready_queue, sorted by arrival to break ties fairly at start
        # This ensures processes arriving at the same initial time are ordered correctly.
        initial_arrivals = [p for p in pending_processes if p.arrival_time <= self.current_time]
        initial_arrivals.sort(key=lambda p_arr: (p_arr.arrival_time, p_arr.pid))
        for p_arr in initial_arrivals:
            self._enqueue_process(p_arr)
            if p_arr in pending_processes: pending_processes.remove(p_arr)


        while len(self.completed_processes) < num_total_processes:
            # Add newly arrived processes to the end of the ready queue
            newly_arrived_this_tick_objs = []
            # Iterate backwards for safe removal from pending_processes
            for p_idx in range(len(pending_processes) - 1, -1, -1):
                p = pending_processes[p_idx]
                if p.arrival_time <= self.current_time:
                    newly_arrived_this_tick_objs.append(p)
                    pending_processes.pop(p_idx)

            # Sort newly arrived by arrival time and PID before adding to ready_queue
            newly_arrived_this_tick_objs.sort(key=lambda item:(item.arrival_time, item.pid))
            for p_arrived in newly_arrived_this_tick_objs:
                 self._enqueue_process(p_arrived)


            # Handle current process: completion or time slice expiry
            if current_process:
                if current_process.remaining_time == 0:
                    # Process completed in the previous tick's execution phase
                    current_process.completion_time = self.current_time # Completion happens at start of this tick
                    self._calculate_metrics(current_process)
                    self.completed_processes.append(current_process)
                    if last_gantt_segment_start_time != -1:
                        self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, self.current_time, "完成")
                    current_process = None
                    current_slice_used = 0
                    last_gantt_segment_start_time = -1
                elif current_slice_used == self.time_quantum: # Time slice expired
                    if last_gantt_segment_start_time != -1:
                         self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, self.current_time, "时间片用尽")
                    self._re_enqueue_process(current_process) # Add to end of ready queue
                    current_process = None
                    current_slice_used = 0
                    last_gantt_segment_start_time = -1

            # If no process is running, pick one from ready queue
            if not current_process and self.ready_queue:
                current_process = self._dequeue_process()
                if current_process:
                    last_gantt_segment_start_time = self.current_time
                    current_slice_used = 0 # Reset slice for new/resumed process
                    if current_process.start_time is None:
                        current_process.start_time = self.current_time
                        current_process.response_time = current_process.start_time - current_process.arrival_time

            # Execute current process for one time unit
            if current_process:
                current_process.remaining_time -= 1
                current_slice_used += 1
                self.total_cpu_busy_time += 1
            elif not pending_processes and not self.ready_queue: # No current, no pending, no ready
                break
            elif pending_processes and not self.ready_queue and not current_process: # CPU Idle, but future processes exist
                 next_arrival_time = min(p.arrival_time for p in pending_processes)
                 if self.current_time < next_arrival_time: # Avoid getting stuck if current_time is already at next_arrival
                    self.current_time = next_arrival_time
                    continue # Skip incrementing time at end of loop, and re-evaluate arrivals
                 # else, if current_time == next_arrival_time, let it increment by 1 to process arrivals in next loop

            self.current_time += 1

        self.completed_processes.sort(key=lambda p: (p.completion_time if p.completion_time is not None else float('inf'), p.pid))


# --- Multilevel Queue Scheduler ---
class MultilevelQueueScheduler(Scheduler):
    def __init__(self, processes: List[Process], queue_configs: List[Dict[str, Any]]):
        super().__init__(processes)
        self.algorithm_name = "多级队列调度 (MLQ)"
        self.queue_configs = queue_configs
        self.num_queues = len(queue_configs)
        # Each queue in self.queues is a list of Process objects
        self.queues: List[List[Process]] = [[] for _ in range(self.num_queues)]

        self.rr_time_quantums: Dict[int, int] = {}
        params_display = []
        for i, qc in enumerate(self.queue_configs):
            q_type = qc.get('algo', 'FCFS').upper()
            params_display.append(f"Q{i}: {q_type}" + (f" (TQ={qc['time_quantum']})" if q_type == 'RR' else ""))
            if q_type == 'RR':
                self.rr_time_quantums[i] = qc.get('time_quantum', 1)
        self.algorithm_params = {"队列配置": ", ".join(params_display)}


    def _enqueue_process_to_assigned_queue(self, process: Process) -> None:
        q_idx = process.assigned_queue_index
        if 0 <= q_idx < self.num_queues:
            if process not in self.queues[q_idx]: # Basic duplicate check
                 self.queues[q_idx].append(process)
                 # If FCFS queue, sort it upon adding
                 if self.queue_configs[q_idx].get('algo', 'FCFS').upper() == 'FCFS':
                     self.queues[q_idx].sort(key=lambda p_item: (p_item.arrival_time, p_item.pid))
        else:
            # This path should ideally not be hit if CLI validates q_idx for process
            print(f"警告: 进程 {process.pid} 被分配到无效队列索引 {q_idx}。将分配到队列 0。")
            if process not in self.queues[0]:
                self.queues[0].append(process)
            process.assigned_queue_index = 0


    def run(self) -> None:
        pending_processes = list(self.processes_master_list)
        if not pending_processes: return

        current_process: Optional[Process] = None
        current_process_slice_used = 0
        last_gantt_segment_start_time = -1

        self.current_time = 0
        if self.processes_master_list:
            self.current_time = min(p.arrival_time for p in self.processes_master_list) if self.processes_master_list else 0

        num_total_processes = len(self.processes_master_list)

        # Initial population, sorted by their assigned queue, then arrival.
        initial_arrivals = [p for p in pending_processes if p.arrival_time <= self.current_time]
        initial_arrivals.sort(key=lambda p_arr: (p_arr.assigned_queue_index, p_arr.arrival_time, p_arr.pid))
        for p_arr in initial_arrivals:
            self._enqueue_process_to_assigned_queue(p_arr)
            if p_arr in pending_processes: pending_processes.remove(p_arr)


        while len(self.completed_processes) < num_total_processes:
            # Add newly arrived processes to their assigned queues
            newly_arrived_this_tick = []
            for p_idx in range(len(pending_processes) -1, -1, -1):
                p = pending_processes[p_idx]
                if p.arrival_time <= self.current_time:
                    newly_arrived_this_tick.append(p)
                    pending_processes.pop(p_idx)

            # Sort by assigned queue, then arrival before enqueuing
            newly_arrived_this_tick.sort(key=lambda p_arr: (p_arr.assigned_queue_index, p_arr.arrival_time, p_arr.pid))
            for p_arrived in newly_arrived_this_tick:
                 self._enqueue_process_to_assigned_queue(p_arrived)

            # Handle current process: completion or (if RR) time slice expiry
            if current_process:
                q_idx_current = current_process.assigned_queue_index
                algo_current = self.queue_configs[q_idx_current].get('algo', 'FCFS').upper()

                if current_process.remaining_time == 0:
                    current_process.completion_time = self.current_time
                    self._calculate_metrics(current_process)
                    self.completed_processes.append(current_process)
                    if last_gantt_segment_start_time != -1:
                        self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, self.current_time, "完成")
                    current_process = None
                    last_gantt_segment_start_time = -1
                    current_process_slice_used = 0
                elif algo_current == 'RR' and current_process_slice_used == self.rr_time_quantums.get(q_idx_current, 1):
                    if last_gantt_segment_start_time != -1:
                        self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, self.current_time, f"Q{q_idx_current} 时间片用尽")
                    self.queues[q_idx_current].append(current_process) # Add to end of its own RR queue
                    current_process = None
                    last_gantt_segment_start_time = -1
                    current_process_slice_used = 0

            # Select next process: iterate through queues by priority (0 is highest)
            # Preemption: If a higher priority queue gets a new process, it should take over.
            # This check is implicitly handled by iterating queues 0 to N each time a process is picked.
            if not current_process:
                for q_idx_select in range(self.num_queues):
                    if self.queues[q_idx_select]:
                        current_process = self.queues[q_idx_select].pop(0) # FIFO from this queue
                        current_process_slice_used = 0 # Reset for new process

                        if current_process:
                            last_gantt_segment_start_time = self.current_time
                            if current_process.start_time is None:
                                current_process.start_time = self.current_time
                                current_process.response_time = current_process.start_time - current_process.arrival_time
                        break # Found a process

            # Execute current process
            if current_process:
                current_process.remaining_time -= 1
                self.total_cpu_busy_time += 1
                q_idx_current_run = current_process.assigned_queue_index
                if self.queue_configs[q_idx_current_run].get('algo','FCFS').upper() == 'RR':
                    current_process_slice_used += 1
            elif not pending_processes and all(not q for q in self.queues): # No current, no pending, all queues empty
                break
            elif pending_processes and all(not q for q in self.queues) and not current_process: # Idle
                next_arrival_time = min(p.arrival_time for p in pending_processes)
                if self.current_time < next_arrival_time:
                    self.current_time = next_arrival_time
                    continue
            self.current_time += 1

        self.completed_processes.sort(key=lambda p: (p.completion_time if p.completion_time is not None else float('inf'), p.pid))


# --- Multilevel Feedback Queue Scheduler ---
class MultilevelFeedbackQueueScheduler(Scheduler):
    def __init__(self, processes: List[Process], queue_time_slices: List[Optional[int]], aging_threshold: Optional[int] = None):
        super().__init__(processes)
        self.algorithm_name = "多级反馈队列调度 (MLFQ)"
        self.num_queues = len(queue_time_slices)
        self.queue_time_slices = queue_time_slices
        # Each queue is a list of Process objects (FIFO within level)
        self.queues: List[List[Process]] = [[] for _ in range(self.num_queues)]
        self.aging_threshold = aging_threshold

        slices_display = []
        for i, s_val in enumerate(self.queue_time_slices):
            slices_display.append(f"Q{i}: " + (f"TQ={s_val}" if s_val is not None else "FCFS"))
        self.algorithm_params = {
            "各级队列时间片/策略": ", ".join(slices_display),
            "老化阈值": aging_threshold if aging_threshold is not None else "未启用"
        }

    def _enqueue_to_feedback_queue(self, process: Process, queue_level: int, is_new_arrival_or_promotion: bool = True) -> None:
        if 0 <= queue_level < self.num_queues:
            # Avoid re-adding if it's already there and not meant to be moved (e.g. from current_process check)
            # This check can be tricky, rely on careful calling.
            process.current_queue_level = queue_level
            if is_new_arrival_or_promotion: # Reset slice only if explicitly new to this queue level
                process.time_in_current_queue_slice = 0
            self.queues[queue_level].append(process)
        else:
            print(f"警告: 尝试将进程 {process.pid} 送入无效反馈队列 {queue_level}。")


    def run(self) -> None:
        pending_processes = list(self.processes_master_list)
        if not pending_processes: return

        current_process: Optional[Process] = None
        last_gantt_segment_start_time = -1

        self.current_time = 0
        if self.processes_master_list:
            self.current_time = min(p.arrival_time for p in self.processes_master_list) if self.processes_master_list else 0

        num_total_processes = len(self.processes_master_list)

        # Initial population into highest priority queue (Q0)
        initial_arrivals = [p for p in pending_processes if p.arrival_time <= self.current_time]
        initial_arrivals.sort(key=lambda p_arr: (p_arr.arrival_time, p_arr.pid))
        for p_arr in initial_arrivals:
            p_arr.last_run_time = self.current_time # For aging, set its entry time to Q0 effectively
            self._enqueue_to_feedback_queue(p_arr, 0, is_new_arrival_or_promotion=True)
            if p_arr in pending_processes: pending_processes.remove(p_arr)


        while len(self.completed_processes) < num_total_processes:
            # 1. Aging: Check for processes to promote (from Q1 upwards)
            if self.aging_threshold is not None:
                for q_idx_aging in range(self.num_queues - 1, 0, -1): # Iterate from lowest non-Q0 up to Q1
                    # Iterate backwards for safe removal from self.queues[q_idx_aging]
                    for p_in_q_idx in range(len(self.queues[q_idx_aging]) - 1, -1, -1):
                        p_to_check = self.queues[q_idx_aging][p_in_q_idx]
                        # last_run_time updated when demoted or run. Here, time since it was last "active".
                        if (self.current_time - p_to_check.last_run_time) >= self.aging_threshold and p_to_check.last_run_time < self.current_time :
                            promoted_p = self.queues[q_idx_aging].pop(p_in_q_idx)
                            promoted_p.last_run_time = self.current_time # Update time for new queue / for next aging cycle
                            self._enqueue_to_feedback_queue(promoted_p, q_idx_aging - 1, is_new_arrival_or_promotion=True)
                            print(f"时间 {self.current_time}: 进程 {promoted_p.pid} 因老化从队列 {q_idx_aging} 提升至队列 {q_idx_aging - 1}")


            # 2. Add newly arrived processes to the highest priority queue (Q0)
            newly_arrived_this_tick = []
            for p_idx_arr in range(len(pending_processes) - 1, -1, -1):
                p = pending_processes[p_idx_arr]
                if p.arrival_time <= self.current_time:
                    newly_arrived_this_tick.append(p)
                    pending_processes.pop(p_idx_arr)

            newly_arrived_this_tick.sort(key=lambda p_arr: (p_arr.arrival_time, p_arr.pid))
            for p_arrived in newly_arrived_this_tick:
                 p_arrived.last_run_time = self.current_time # Set for aging
                 self._enqueue_to_feedback_queue(p_arrived, 0, is_new_arrival_or_promotion=True)


            # 3. Handle current process (if any) and check for preemption by higher queue
            if current_process:
                preempted_by_higher_queue = False
                for higher_q_idx in range(current_process.current_queue_level):
                    if self.queues[higher_q_idx]: # If a higher priority queue is now non-empty
                        if last_gantt_segment_start_time != -1:
                            self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, self.current_time, f"被Q{higher_q_idx}抢占")
                        current_process.last_run_time = self.current_time
                        self._enqueue_to_feedback_queue(current_process, current_process.current_queue_level, is_new_arrival_or_promotion=False) # Put back, don't reset slice
                        current_process = None
                        last_gantt_segment_start_time = -1
                        preempted_by_higher_queue = True
                        break
                if preempted_by_higher_queue:
                    pass # Handled, current_process is None
                # No preemption from higher queue, continue with current_process logic for completion/demotion
                elif current_process.remaining_time == 0:
                    current_process.completion_time = self.current_time
                    self._calculate_metrics(current_process)
                    self.completed_processes.append(current_process)
                    if last_gantt_segment_start_time != -1:
                        self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, self.current_time, "完成")
                    current_process = None
                    last_gantt_segment_start_time = -1
                else: # Not completed, check time slice for demotion
                    current_q_level = current_process.current_queue_level
                    current_q_time_slice = self.queue_time_slices[current_q_level]
                    if current_q_time_slice is not None and current_process.time_in_current_queue_slice == current_q_time_slice:
                        if last_gantt_segment_start_time != -1:
                             self._add_gantt_entry(current_process.pid, last_gantt_segment_start_time, self.current_time, f"Q{current_q_level} 时间片用尽")

                        next_q_level = current_q_level + 1
                        current_process.last_run_time = self.current_time # Update before demotion/re-queue
                        if next_q_level < self.num_queues: # Demote
                            self._enqueue_to_feedback_queue(current_process, next_q_level, is_new_arrival_or_promotion=True)
                            # print(f"时间 {self.current_time}: 进程 {current_process.pid} 从队列 {current_q_level} 降至队列 {next_q_level}") # Already printed by _add_gantt_entry
                        else: # Last queue, re-add to end of same queue
                            self._enqueue_to_feedback_queue(current_process, current_q_level, is_new_arrival_or_promotion=True) # Reset slice for last level too
                        current_process = None
                        last_gantt_segment_start_time = -1

            # 4. Select next process from highest priority non-empty queue if no process is running
            if not current_process:
                for q_idx_select_fb in range(self.num_queues):
                    if self.queues[q_idx_select_fb]:
                        current_process = self.queues[q_idx_select_fb].pop(0) # FIFO from the selected queue
                        # current_process.time_in_current_queue_slice = 0 # Reset slice counter - ALREADY DONE in _enqueue if promoted/new
                        if current_process:
                            last_gantt_segment_start_time = self.current_time
                            if current_process.start_time is None:
                                current_process.start_time = self.current_time
                                current_process.response_time = current_process.start_time - current_process.arrival_time
                        break

            # 5. Execute current process for one time unit
            if current_process:
                current_process.remaining_time -= 1
                current_process.time_in_current_queue_slice += 1
                current_process.last_run_time = self.current_time # Update last time it was active
                self.total_cpu_busy_time += 1
            elif not pending_processes and all(not q for q in self.queues): # No current, no pending, all queues empty
                break
            elif pending_processes and all(not q for q in self.queues) and not current_process: # Idle
                next_arrival_time = min(p.arrival_time for p in pending_processes)
                if self.current_time < next_arrival_time:
                    self.current_time = next_arrival_time
                    continue
            self.current_time += 1

        self.completed_processes.sort(key=lambda p: (p.completion_time if p.completion_time is not None else float('inf'), p.pid))


# --- Utility Functions ---
def get_process_data_manually(needs_priority_for_any_algo: bool) -> List[Process]:
    processes = []
    num_processes_str = input("请输入进程数量: ")
    while not num_processes_str.isdigit() or int(num_processes_str) <= 0:
        num_processes_str = input("无效输入。请输入一个正整数作为进程数量: ")
    num_processes = int(num_processes_str)

    has_priority_input = False
    if needs_priority_for_any_algo: # Only ask if any selected algo might need it.
        ask_priority_choice = input("是否为进程定义优先级 (y/n)? (某些调度算法需要): ").lower()
        if ask_priority_choice == 'y':
            has_priority_input = True
            print("注意: 优先级数值越小，优先级越高。")

    for i in range(num_processes):
        print(f"\n输入进程 {i+1} 的信息:")
        pid = input(f"  进程ID (例如 P{i+1}): ") or f"P{i+1}"

        arrival_time = -1
        while arrival_time < 0:
            try:
                arrival_time_str = input("  到达时间 (非负整数): ")
                arrival_time = int(arrival_time_str)
                if arrival_time < 0: print("到达时间不能为负。")
            except ValueError:
                print("无效输入，请输入一个整数。")

        run_time = 0
        while run_time <= 0:
            try:
                run_time_str = input("  运行时间 (正整数): ")
                run_time = int(run_time_str)
                if run_time <= 0: print("运行时间必须为正数。")
            except ValueError:
                print("无效输入，请输入一个整数。")

        priority: Optional[int] = None
        if has_priority_input:
            while True:
                try:
                    priority_str = input("  优先级 (整数, 建议 >0, 或留空不设置): ")
                    if not priority_str:
                        priority = None
                        break
                    priority = int(priority_str)
                    # Allowing 0 or negative as priority, though conventionally positive.
                    break
                except ValueError:
                    print("无效输入。请输入一个整数或留空。")
        processes.append(Process(pid, arrival_time, run_time, priority))
    return processes

def generate_random_processes(num_processes: int, max_arrival: int, max_run_time: int, include_priority: bool) -> List[Process]:
    processes = []
    for i in range(num_processes):
        pid = f"P{i+1}"
        arrival_time = random.randint(0, max_arrival)
        run_time = random.randint(1, max_run_time)
        priority = random.randint(1, 10) if include_priority else None
        processes.append(Process(pid, arrival_time, run_time, priority))
    return processes

def check_priorities_defined(processes: List[Process]) -> bool:
    return all(p.priority is not None for p in processes)

# --- Main CLI ---
def main_cli():
    print("欢迎使用操作系统调度算法可视化工具!")

    algorithms_needing_explicit_priority = ['4', '5']


    active_processes: List[Process] = []
    while True: # Main loop for process input and simulation
        if not active_processes:
            print("\n选择进程数据输入方式:")
            print("1. 手动输入进程数据")
            print("2. 随机生成进程数据")
            print("3. 退出程序")
            choice = input("请输入选项 (1-3): ")

            if choice == '1':
                active_processes = get_process_data_manually(True)
            elif choice == '2':
                while True:
                    try:
                        num_p_str = input("要生成的进程数量 (正整数): ")
                        max_a_str = input("最大到达时间 (非负整数): ")
                        max_r_str = input("最大运行时间 (正整数): ")
                        if not (num_p_str.isdigit() and max_a_str.isdigit() and max_r_str.isdigit()):
                            raise ValueError("所有输入必须是数字。")
                        num_p = int(num_p_str)
                        max_a = int(max_a_str)
                        max_r = int(max_r_str)

                        if not (num_p > 0 and max_a >= 0 and max_r > 0):
                            print("数量需为正，到达时间需非负，运行时间需为正。")
                            continue

                        priority_choice_random = input("为随机进程生成优先级? (y/n): ").lower()
                        gen_priority = priority_choice_random == 'y'
                        active_processes = generate_random_processes(num_p, max_a, max_r, gen_priority)
                        break
                    except ValueError as e:
                        print(f"无效输入: {e}。请重试。")
            elif choice == '3':
                print("感谢使用，再见!")
                return
            else:
                print("无效选项，请重新输入。")
                continue
        if not active_processes: continue

        temp_scheduler_for_print = Scheduler(active_processes)
        temp_scheduler_for_print.print_initial_processes()


        while True:
            print("\n请选择要模拟的调度算法:")
            print("1. FCFS (先来先服务)")
            print("2. 非抢占式SJF (短作业优先)")
            print("3. 抢占式SJF (SRTF, 最短剩余时间优先)")
            print("4. 非抢占式优先级调度")
            print("5. 抢占式优先级调度")
            print("6. 时间片轮转 (Round Robin, RR)")
            print("7. 多级队列调度 (Multilevel Queue)")
            print("8. 多级反馈队列调度 (Multilevel Feedback Queue)")
            print("9. 输入新的进程数据")
            print("0. 退出程序")
            algo_choice = input("请输入选项 (0-9): ")

            scheduler: Optional[Scheduler] = None

            if algo_choice == '0':
                print("感谢使用，再见!")
                return
            if algo_choice == '9':
                active_processes = []
                break

            if algo_choice in algorithms_needing_explicit_priority and not check_priorities_defined(active_processes):
                print("\n错误: 所选算法需要为所有进程定义优先级。")
                fix_priority_choice = input("是否现在为所有进程补充/修改优先级? (y/n): ").lower()
                if fix_priority_choice == 'y':
                    print("为每个进程输入/更新优先级 (数值越小，优先级越高):")
                    for p_obj in active_processes:
                        while True:
                            try:
                                p_val_str = input(f"  进程 {p_obj.pid} (当前优先级: {p_obj.priority if p_obj.priority is not None else 'N/A'}) 新的优先级 (整数): ")
                                if not p_val_str and p_obj.priority is not None:
                                    break
                                p_obj.priority = int(p_val_str)
                                break
                            except ValueError:
                                print("无效输入。请输入一个整数。")
                    if not check_priorities_defined(active_processes):
                        print("仍有进程未定义有效优先级。请选择其他算法或重新定义优先级。")
                        continue
                else:
                    print("请选择其他算法或重新输入进程数据并定义优先级。")
                    continue

            if algo_choice == '1': scheduler = FCFS(active_processes)
            elif algo_choice == '2': scheduler = NonPreemptiveSJF(active_processes)
            elif algo_choice == '3': scheduler = SRTF(active_processes)
            elif algo_choice == '4': scheduler = NonPreemptivePriority(active_processes)
            elif algo_choice == '5': scheduler = PreemptivePriority(active_processes)
            elif algo_choice == '6':
                while True:
                    try:
                        quantum_str = input("请输入时间片大小 (正整数): ")
                        quantum = int(quantum_str)
                        if quantum <= 0: raise ValueError("时间片必须为正。")
                        scheduler = RoundRobin(active_processes, quantum)
                        break
                    except ValueError as e: print(f"无效输入: {e}")
            elif algo_choice == '7':
                print("\n配置多级队列调度:")
                num_q_str = input("请输入队列数量 (至少1): ")
                if not num_q_str.isdigit() or int(num_q_str) < 1:
                    print("无效队列数量，默认为2。")
                    num_q = 2
                else: num_q = int(num_q_str)

                q_configs_mlq = []
                for i in range(num_q):
                    print(f"\n配置队列 {i} (优先级从高到低 0, 1, ...):")
                    q_algo_mlq = input(f"  队列 {i} 的调度算法 (FCFS, RR): ").upper()
                    config_mlq: Dict[str, Any] = {'algo': q_algo_mlq}
                    if q_algo_mlq == 'RR':
                        while True:
                            try:
                                qt_str = input(f"  队列 {i} (RR) 的时间片大小 (正整数): ")
                                qt_mlq = int(qt_str)
                                if qt_mlq <= 0: raise ValueError("时间片必须为正。")
                                config_mlq['time_quantum'] = qt_mlq
                                break
                            except ValueError as e: print(f"无效输入: {e}")
                    elif q_algo_mlq not in ['FCFS']:
                        print(f"不支持的算法 {q_algo_mlq} for MLQ sub-queue, 默认为 FCFS。")
                        config_mlq['algo'] = 'FCFS'
                    q_configs_mlq.append(config_mlq)

                print("\n为进程分配队列 (0 是最高优先级队列):")
                for p_obj in active_processes:
                    while True:
                        try:
                            q_idx_str = input(f"  进程 {p_obj.pid} (到达:{p_obj.arrival_time}, 运行:{p_obj.run_time}) 分配到队列索引 (0-{num_q-1}): ")
                            q_idx_mlq = int(q_idx_str)
                            if not (0 <= q_idx_mlq < num_q):
                                raise ValueError(f"队列索引必须在 0 到 {num_q-1} 之间。")
                            p_obj.assigned_queue_index = q_idx_mlq
                            break
                        except ValueError as e: print(f"无效输入: {e}")
                scheduler = MultilevelQueueScheduler(active_processes, q_configs_mlq)

            elif algo_choice == '8':
                print("\n配置多级反馈队列调度:")
                num_f_q_str = input("请输入队列数量 (至少1): ")
                if not num_f_q_str.isdigit() or int(num_f_q_str) < 1:
                    print("无效队列数量，默认为2。")
                    num_f_q = 2
                else: num_f_q = int(num_f_q_str)

                q_time_slices_mlfq:List[Optional[int]] = []
                print("优先级越高的队列时间片越小。最后一个队列可以是FCFS (时间片留空)。")
                for i in range(num_f_q):
                    while True:
                        try:
                            slice_str = input(f"  队列 {i} 的时间片大小 (正整数, 或留空表示FCFS - 仅最低优先级队列): ")
                            if not slice_str and i == num_f_q - 1:
                                q_time_slices_mlfq.append(None) # FCFS for last queue
                                break
                            elif not slice_str:
                                print("只有最低优先级队列的时间片可以留空 (FCFS)。")
                                continue
                            time_slice_mlfq = int(slice_str)
                            if time_slice_mlfq <= 0: raise ValueError("时间片必须为正整数。")
                            q_time_slices_mlfq.append(time_slice_mlfq)
                            break
                        except ValueError as e: print(f"无效输入: {e}")

                aging_choice_mlfq = input("是否启用老化机制? (y/n): ").lower()
                aging_thresh_mlfq: Optional[int] = None
                if aging_choice_mlfq == 'y':
                    while True:
                        try:
                            thresh_str = input("请输入老化阈值 (正整数, 等待超过此时间则提升): ")
                            aging_thresh_mlfq = int(thresh_str)
                            if aging_thresh_mlfq <=0: raise ValueError("老化阈值必须为正。")
                            break
                        except ValueError as e: print(f"无效输入: {e}")
                scheduler = MultilevelFeedbackQueueScheduler(active_processes, q_time_slices_mlfq, aging_thresh_mlfq)

            else:
                if algo_choice not in ['0', '9']:
                    print("无效算法选项。请重新输入。")
                continue

            if scheduler:
                print(f"\n--- 开始执行 {scheduler.algorithm_name} 调度 ---")
                scheduler.run()
                scheduler.print_results()
                try:
                    scheduler.plot_gantt_and_metrics()
                except ImportError:
                     print("绘图库 Plotly 或 Pandas 未安装。请运行 'pip install plotly pandas kaleido'。")
                except Exception as e:
                    print(f"生成图表时发生错误: {e}")
                    print("确保已安装 Plotly, Pandas (pip install plotly pandas) 和 Kaleido (pip install kaleido)。")


if __name__ == "__main__":
    main_cli()