from typing import List, Optional
import random
import plotly.graph_objects as go


class Process:
    def __init__(self, pid: int, arrival_time: int, run_time: int, priority: Optional[int] = None):
        self.pid = pid
        self.arrival_time = arrival_time
        self.run_time = run_time
        self.priority = priority
        self.remaining_time = run_time
        self.start_time = None
        self.completion_time = None
        self.waiting_time = None
        self.turnaround_time = None
        self.response_time = None

    def calculate_metrics(self):
        if self.completion_time is not None and self.arrival_time is not None:
            self.turnaround_time = self.completion_time - self.arrival_time
            self.waiting_time = self.turnaround_time - self.run_time
            if self.start_time is not None:
                self.response_time = self.start_time - self.arrival_time


class Scheduler:
    def __init__(self, processes: List[Process]):
        self.processes = sorted(processes, key=lambda x: x.arrival_time)
        self.ready_queue = []
        self.current_time = 0
        self.completed_processes = []

    def update_ready_queue(self):
        for process in self.processes[:]:
            if process.arrival_time <= self.current_time:
                self.ready_queue.append(process)
                self.processes.remove(process)

    def schedule(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def print_process_details(self):
        print(f"{'PID':<5} {'Arrival':<10} {'Run Time':<10} {'Priority':<10} "
              f"{'Start':<10} {'Completion':<12} {'Waiting':<10} {'Turnaround':<13} {'Response':<10}")
        for process in self.processes + self.ready_queue + self.completed_processes:
            start_str = str(process.start_time) if process.start_time is not None else '-'
            completion_str = str(process.completion_time) if process.completion_time is not None else '-'
            waiting_str = str(process.waiting_time) if process.waiting_time is not None else '-'
            turnaround_str = str(process.turnaround_time) if process.turnaround_time is not None else '-'
            response_str = str(process.response_time) if process.response_time is not None else '-'
            priority_str = str(process.priority) if process.priority is not None else '-'
            print(f"{process.pid:<5} {process.arrival_time:<10} {process.run_time:<10} {priority_str:<10} "
                  f"{start_str:<10} {completion_str:<12} {waiting_str:<10} {turnaround_str:<13} {response_str:<10}")

    def print_execution_sequence(self, execution_sequence):
        print("\nExecution Sequence:")
        for time_range, pid, action in execution_sequence:
            print(f"时间 [{time_range[0]}-{time_range[1]}]: 进程 [{pid}] ({action})")

    def calculate_performance_metrics(self):
        total_turnaround_time = sum(p.turnaround_time for p in self.completed_processes)
        total_waiting_time = sum(p.waiting_time for p in self.completed_processes)
        total_response_time = sum(p.response_time for p in self.completed_processes)
        cpu_utilization = (sum(p.run_time for p in self.completed_processes) / self.current_time) * 100
        throughput = len(self.completed_processes) / self.current_time
        avg_turnaround_time = total_turnaround_time / len(self.completed_processes) if self.completed_processes else 0
        avg_waiting_time = total_waiting_time / len(self.completed_processes) if self.completed_processes else 0
        avg_response_time = total_response_time / len(self.completed_processes) if self.completed_processes else 0

        print("\nPerformance Metrics:")
        print(f"CPU Utilization: {cpu_utilization:.2f}%")
        print(f"Throughput: {throughput:.2f} processes/unit time")
        print(f"Average Turnaround Time: {avg_turnaround_time:.2f}")
        print(f"Average Waiting Time: {avg_waiting_time:.2f}")
        print(f"Average Response Time: {avg_response_time:.2f}")

    def visualize_gantt_chart(self, execution_sequence, algorithm_name):
        fig = go.Figure()

        colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa', '#ffa15a']
        color_map = {seq[1]: colors[i % len(colors)] for i, seq in enumerate(execution_sequence)}

        tasks = [f"进程 {seq[1]}" for seq in execution_sequence]
        starts = [seq[0][0] for seq in execution_sequence]
        durations = [seq[0][1] - seq[0][0] for seq in execution_sequence]

        fig.add_trace(go.Bar(
            y=tasks,
            x=durations,
            base=starts,
            orientation='h',
            marker_color=[color_map[t.split()[1]] for t in tasks],
            text=[f"{seq[2]} [{seq[0][0]}-{seq[0][1]}]" for seq in execution_sequence],
            hoverinfo='text'
        ))

        fig.update_layout(
            title=f"{algorithm_name} 调度甘特图",
            xaxis_title="时间",
            yaxis_title="进程",
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis=dict(categoryorder='total descending'),
            bargap=0.2,
            font=dict(size=14, family="SimHei"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig.show()

        # Performance metrics table
        performance_metrics = [
            ("CPU Utilization", f"{(sum(p.run_time for p in self.completed_processes) / self.current_time) * 100:.2f}%"),
            ("Throughput", f"{len(self.completed_processes) / self.current_time:.2f} processes/unit time"),
            ("Average Turnaround Time", f"{sum(p.turnaround_time for p in self.completed_processes) / len(self.completed_processes):.2f}" if self.completed_processes else "N/A"),
            ("Average Waiting Time", f"{sum(p.waiting_time for p in self.completed_processes) / len(self.completed_processes):.2f}" if self.completed_processes else "N/A"),
            ("Average Response Time", f"{sum(p.response_time for p in self.completed_processes) / len(self.completed_processes):.2f}" if self.completed_processes else "N/A")
        ]

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=["性能指标", "值"], fill_color='paleturquoise', align='left'),
            cells=dict(values=list(zip(*performance_metrics)), fill_color='lavender', align='left')
        )])

        fig_table.update_layout(title_text="系统整体性能指标", font=dict(size=14, family="SimHei"))

        fig_table.show()


class FCFSScheduler(Scheduler):
    def schedule(self):
        execution_sequence = []
        while self.processes or self.ready_queue:
            self.update_ready_queue()
            if self.ready_queue:
                current_process = self.ready_queue.pop(0)
                if current_process.start_time is None:
                    current_process.start_time = self.current_time
                    current_process.response_time = self.current_time - current_process.arrival_time
                end_time = self.current_time + current_process.remaining_time
                execution_sequence.append(((self.current_time, end_time), current_process.pid, "执行"))
                self.current_time = end_time
                current_process.completion_time = self.current_time
                current_process.calculate_metrics()
                self.completed_processes.append(current_process)
            else:
                self.current_time += 1

        return execution_sequence


def manual_input():
    num_processes = int(input("请输入进程数量: "))
    processes = []
    for i in range(num_processes):
        pid = i + 1
        arrival_time = int(input(f"请输入进程 {pid} 的到达时间: "))
        run_time = int(input(f"请输入进程 {pid} 的运行时间: "))
        priority = input(f"请输入进程 {pid} 的优先级 (可选): ")
        priority = int(priority) if priority else None
        processes.append(Process(pid, arrival_time, run_time, priority))
    return processes


def generate_random_processes(num_processes, max_arrival_time, max_run_time):
    processes = []
    for i in range(num_processes):
        pid = i + 1
        arrival_time = random.randint(0, max_arrival_time)
        run_time = random.randint(1, max_run_time)
        priority = random.randint(1, 10)  # Random priority between 1 and 10
        processes.append(Process(pid, arrival_time, run_time, priority))
    return processes


if __name__ == "__main__":
    print("欢迎使用操作系统调度算法可视化工具！")
    input_method = input("请选择进程数据输入方式（手动/随机）: ").strip().lower()
    if input_method == "手动":
        processes = manual_input()
    elif input_method == "随机":
        num_processes = int(input("请输入生成进程的数量: "))
        max_arrival_time = int(input("请输入最大到达时间: "))
        max_run_time = int(input("请输入最大运行时间: "))
        processes = generate_random_processes(num_processes, max_arrival_time, max_run_time)
    else:
        print("无效的输入，请选择 '手动' 或 '随机'")
        exit(1)

    scheduler_type = input("请选择要模拟的调度算法（FCFS, SJF非抢占式, SRTF, 优先级非抢占式, 优先级抢占式, RR, 多级队列, 多级反馈队列）: ").strip().upper()
    if scheduler_type == "FCFS":
        scheduler = FCFSScheduler(processes)
        execution_sequence = scheduler.schedule()
        scheduler.print_process_details()
        scheduler.print_execution_sequence(execution_sequence)
        scheduler.calculate_performance_metrics()
        scheduler.visualize_gantt_chart(execution_sequence, "FCFS")
    else:
        print("目前仅支持 FCFS 算法，请稍后选择其他算法。")



