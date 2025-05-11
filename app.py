import os
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext, filedialog
from threading import Thread
from lab1 import TextToGraph  # 请确保这个模块存在

# 这是一个GUI界面，实现了类GraphApp
# 详情类逻辑请到文件lab1中查看类TextToGraph
# git push origin main
class GraphApp:
    def __init__(self, root):
        self.text = None
        self.root = root
        self.root.title("🔍 有向图文本分析工具")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        self.graph_builder = None
        self.dot_file_path = "./graph/directed_graph.dot"

        os.makedirs("./graph", exist_ok=True)

        self.file_path = ""
        self.all_words = []
        self.root_word = ""
        self.walk_thread = None

        self.create_widgets()

    def create_widgets(self):
        title = tk.Label(self.root, text="有向图文本分析工具", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333")
        title.pack(pady=10)

        # 上传按钮
        upload_btn = tk.Button(self.root, text="📁 上传文本文件", width=25, font=("Helvetica", 12),
                               bg="#5cb85c", fg="white", relief="raised", bd=2,
                               command=self.upload_file)
        upload_btn.pack(pady=(0, 10))

        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=10)

        options = [
            ("🔗 桥接词查询", self.query_bridge_words),
            ("📝 生成新文本", self.generate_new_text),
            ("📍 计算最短路径", self.shortest_path),
            ("🎲 随机游走", self.random_walk),
            ("📊 PageRank计算", self.page_rank),
            ("❌ 退出程序", self.root.quit)
        ]

        for i, (text, command) in enumerate(options):
            btn = tk.Button(btn_frame, text=text, width=20, font=("Helvetica", 12),
                            bg="#4a90e2", fg="white", relief="groove", bd=2,
                            command=command)
            btn.grid(row=i // 2, column=i % 2, padx=20, pady=8)

        self.output_text = scrolledtext.ScrolledText(self.root, width=90, height=20,
                                                     font=("Courier New", 11),
                                                     bg="#ffffff", fg="#333333", borderwidth=2, relief="sunken")
        self.output_text.pack(pady=10)

    def upload_file(self):
        file_path = filedialog.askopenfilename(title="选择文本文件", filetypes=[("Text files", "*.txt")])
        if not file_path:
            return
        self.file_path = file_path
        self.graph_builder = TextToGraph()
        self.output_text.insert(tk.END, f"[上传文件] 已选择文件：{file_path}\n")
        # 读取文件内容并显示
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
                self.text = file_content
        except Exception as e:
            messagebox.showerror("错误", f"读取文件时出错: {e}")
            return
        # 重新生成图并初始化
        self.all_words = self.graph_builder.show_directed_graph(self.file_path, self.dot_file_path, "")
        if self.all_words:
            self.root_word = self.all_words[0]

    def check_file_loaded(self):
        if not self.file_path:
            messagebox.showerror("错误", "请先上传文本文件。")
            return False
        return True

    def query_bridge_words(self):
        if not self.check_file_loaded():
            return
        input_text = simpledialog.askstring("桥接词查询", "请输入两个词语（空格分隔）:")
        if not input_text:
            return
        words = input_text.strip().split()
        if len(words) != 2:
            messagebox.showerror("错误", "请输入两个词语")
            return
        result = self.graph_builder.query_bridge_words(words[0].lower(), words[1].lower())
        self.output_text.insert(tk.END, f"[桥接词查询] {words[0]} → {words[1]}：{result}\n")

    def generate_new_text(self):
        if not self.check_file_loaded():
            return
        input_text = simpledialog.askstring("生成新文本", "请输入文本：")
        if not input_text:
            return
        result = self.graph_builder.generate_new_text(input_text)
        self.output_text.insert(tk.END, f"[生成文本] {result}\n")

    def shortest_path(self):
        if not self.check_file_loaded():
            return
        input_text = simpledialog.askstring("最短路径", "请输入一个或两个词语（空格分隔）:")
        if not input_text:
            return
        words = input_text.strip().split()

        if len(words) == 1:
            for i, word in enumerate(self.all_words[1:], 1):
                path = self.graph_builder.calc_shortest_path(words[0].lower(), word.lower(), i, self.root_word)
                self.output_text.insert(tk.END, f"[最短路径] {words[0]} → {word}：{path}\n")
        elif len(words) == 2:
            path = self.graph_builder.calc_shortest_path(words[0].lower(), words[1].lower(), 1, self.root_word)
            self.output_text.insert(tk.END, f"[最短路径] {words[0]} → {words[1]}：{path}\n")
        else:
            messagebox.showerror("错误", "请输入1或2个词语")

    def random_walk(self):
        if not self.check_file_loaded():
            return
        if self.walk_thread and self.walk_thread.is_alive():
            messagebox.showinfo("提示", "随机游走已在进行中")
            return

        self.output_text.insert(tk.END, "[随机游走] 开始...\n")
        self.graph_builder.stop_walk.clear()

        def run_walk():
            self.graph_builder.random_walk()
            self.output_text.insert(tk.END, "[随机游走] 结束。\n")

        self.walk_thread = Thread(target=run_walk)
        self.walk_thread.start()
        messagebox.showinfo("提示", "点击“确定”后停止游走")
        self.graph_builder.stop_walk.set()

    def page_rank(self):
        if not self.check_file_loaded():
            return
        pr_scores = self.graph_builder.page_rank(text=self.text)
        word = simpledialog.askstring("PageRank查询", "请输入要查询的单词（可留空查看所有）：")

        if word:
            word = word.strip().lower()
            score = pr_scores.get(word)
            if score:
                self.output_text.insert(tk.END, f"[PageRank] {word}: {score:.6f}\n")
            else:
                self.output_text.insert(tk.END, f"[PageRank] {word} 不在图中。\n")
        else:
            self.output_text.insert(tk.END, "[PageRank] 全部结果：\n")
            for w, s in sorted(pr_scores.items(), key=lambda item: item[1], reverse=True):
                self.output_text.insert(tk.END, f"{w}: {s:.6f}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()
    print("Hello Text2Graph")
#   pass

