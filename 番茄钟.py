import tkinter as tk
from tkinter import messagebox
import time
import json


class PomodoroApp:
    def __init__(self, root):
        self.root = root
        self.root.title("番茄工作法应用")
        self.root.geometry("400x350")

        self.timer_running = False
        self.focus_time = 25 * 60
        self.break_time = 5 * 60
        self.current_time = self.focus_time
        self.task_count = 0
        self.total_focus_time = 0
        self.task_category = tk.StringVar()
        self.task_category.set("工作")
        self.task_categories = {}

        self.load_data()

        self.timer_label = tk.Label(root, text=self.format_time(self.current_time), font=("Arial", 30))
        self.timer_label.pack(pady=20)

        self.start_button = tk.Button(root, text="开始", command=self.start_timer)
        self.start_button.pack(pady=10)

        self.pause_button = tk.Button(root, text="暂停", command=self.pause_timer, state=tk.DISABLED)
        self.pause_button.pack(pady=10)

        self.reset_button = tk.Button(root, text="重置", command=self.reset_timer, state=tk.DISABLED)
        self.reset_button.pack(pady=10)

        self.customize_frame = tk.Frame(root)
        self.customize_frame.pack(pady=10)

        tk.Label(self.customize_frame, text="专注时间 (分钟):").grid(row=0, column=0)
        self.focus_entry = tk.Entry(self.customize_frame)
        self.focus_entry.insert(0, "25")
        self.focus_entry.grid(row=0, column=1)

        tk.Label(self.customize_frame, text="休息时间 (分钟):").grid(row=1, column=0)
        self.break_entry = tk.Entry(self.customize_frame)
        self.break_entry.insert(0, "5")
        self.break_entry.grid(row=1, column=1)

        self.customize_button = tk.Button(self.customize_frame, text="自定义时间", command=self.customize_time)
        self.customize_button.grid(row=2, columnspan=2)

        tk.Label(root, text="任务分类:").pack()
        tk.OptionMenu(root, self.task_category, "工作", "学习", "运动").pack()

        self.stats_button = tk.Button(root, text="查看统计", command=self.show_stats)
        self.stats_button.pack(pady=20)

    def load_data(self):
        try:
            with open("pomodoro_stats.json", "r") as file:
                data = json.load(file)
                self.task_count = data.get("task_count", 0)
                self.total_focus_time = data.get("total_focus_time", 0)
                self.task_categories = data.get("task_categories", {})
        except FileNotFoundError:
            pass

    def save_data(self):
        data = {
            "task_count": self.task_count,
            "total_focus_time": self.total_focus_time,
            "task_categories": self.task_categories
        }
        with open("pomodoro_stats.json", "w") as file:
            json.dump(data, file)

    def format_time(self, seconds):
        minutes, secs = divmod(seconds, 60)
        return f"{minutes:02d}:{secs:02d}"

    def start_timer(self):
        if not self.timer_running:
            self.timer_running = True
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.update_timer()

    def pause_timer(self):
        if self.timer_running:
            self.timer_running = False
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)

    def reset_timer(self):
        self.timer_running = False
        self.current_time = self.focus_time
        self.timer_label.config(text=self.format_time(self.current_time))
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)

    def update_timer(self):
        if self.timer_running:
            if self.current_time > 0:
                self.current_time -= 1
                self.timer_label.config(text=self.format_time(self.current_time))
                self.root.after(1000, self.update_timer)
            else:
                if self.current_time == 0:
                    if self.current_time == self.focus_time:
                        self.task_count += 1
                        self.total_focus_time += self.focus_time
                        category = self.task_category.get()
                        if category in self.task_categories:
                            self.task_categories[category] += self.focus_time
                        else:
                            self.task_categories[category] = self.focus_time
                        self.save_data()
                        messagebox.showinfo("提示", "专注时间结束，开始休息！")
                        self.current_time = self.break_time
                    else:
                        messagebox.showinfo("提示", "休息时间结束，开始新的专注时间！")
                        self.current_time = self.focus_time
                    self.timer_label.config(text=self.format_time(self.current_time))
                    self.root.after(1000, self.update_timer)

    def customize_time(self):
        try:
            focus_minutes = int(self.focus_entry.get())
            break_minutes = int(self.break_entry.get())
            self.focus_time = focus_minutes * 60
            self.break_time = break_minutes * 60
            self.current_time = self.focus_time
            self.timer_label.config(text=self.format_time(self.current_time))
            messagebox.showinfo("提示", "时间已自定义！")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数！")

    def show_stats(self):
        stats = f"完成任务数: {self.task_count}\n"
        stats += f"总专注时长: {self.format_time(self.total_focus_time)}\n"
        stats += "各任务分类专注时长:\n"
        for category, time in self.task_categories.items():
            stats += f"{category}: {self.format_time(time)}\n"
        messagebox.showinfo("统计信息", stats)


if __name__ == "__main__":
    root = tk.Tk()
    app = PomodoroApp(root)
    root.mainloop()
    