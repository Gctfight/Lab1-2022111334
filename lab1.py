# -*- coding: utf-8 -*-

import re
import random
import sys
import time
from collections import defaultdict
from threading import Thread, Event
from typing import Dict, List, Set, Tuple, Optional
from graphviz import Digraph
import os
from tqdm import tqdm
from collections import defaultdict
import math

os.environ["PATH"] += os.pathsep + "F:\\software\\Graphviz\\bin"


class TextToGraph:
    def __init__(self):
        # 某个词对另一个词的连续出现次数
        self.directed_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.stop_walk = Event()
        self.file_path = None

    def add_node(self, node: str) -> None:
        if node not in self.directed_graph:
            self.directed_graph[node] = defaultdict(int)

    def add_edge(self, source: str, destination: str) -> None:
        self.directed_graph[source][destination] += 1

    def build_directed_graph(self, file_path: str) -> List[str]:
        content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                content.append(line.strip() + " ")

        processed_content = re.sub(r'[^a-zA-Z\n\r]', ' ', ''.join(content)).lower()
        words = [word for word in re.split(r'\s+', processed_content) if word]

        for i in tqdm(range(len(words) - 1), desc="构建图中"):
            current_word = words[i]
            next_word = words[i + 1]
            self.add_node(current_word)
            self.add_node(next_word)
            self.add_edge(current_word, next_word)

        return words

    def create_dot_file(self, dot_file_path: str, shortest_paths: Optional[List[List]] = None,
                        root: str = None) -> None:
        dot = Digraph(engine="sfdp")  # 比 dot 快很多

        # 设置根节点样式
        if root:
            dot.node(root, style='filled', fillcolor='lightgray')

        all_edges = []
        for vertex, edges in self.directed_graph.items():
            for destination, weight in edges.items():
                all_edges.append((vertex, destination, weight))

        print(f"图节点和边的数量：{len(self.directed_graph)}个节点，{len(all_edges)}条边")
        print("正在生成图的边...")

        for vertex, destination, weight in tqdm(all_edges, desc="绘制边", unit="条"):
            dot.edge(vertex, destination, label=str(weight))

        # 标记最短路径
        if shortest_paths:
            colors = ["red", "green", "blue", "yellow", "orange", "purple"]
            for i, (path, length) in enumerate(shortest_paths):
                color = colors[i % len(colors)]
                for j in range(len(path) - 1):
                    dot.edge(path[j], path[j + 1], color=color, penwidth='2')

        # 保存DOT文件并渲染为PNG
        base_name = os.path.splitext(dot_file_path)[0]
        print("正在渲染图像...")
        dot.render(base_name, format='png', cleanup=True)
        print(f"有向图生成成功: {base_name}.png")

    def show_directed_graph(self, file_path: str, dot_file_path: str, image_file_path: str) -> List[str]:
        print("开始生成有向图")
        words = self.build_directed_graph(file_path)
        self.create_dot_file(dot_file_path, None, words[0])
        return words

    def query_bridge_words(self, start: str, end: str) -> Optional[Set[str]]:
        # ✅ 1. 判断是否为空字符串
        if not start or not end:
            if not start:
                print("单词不能为空")
            if not end:
                print("单词不能为空")
            return None
        # ✅ 检查是否为小写单词，只允许 a-z
        if not re.fullmatch(r"[a-z]+", start):
            print(f"“{start}” 不是合法的小写单词")
            return None
        if not re.fullmatch(r"[a-z]+", end):
            print(f"“{end}” 不是合法的小写单词")
            return None
        bridge_words = set()
        print_result = True
        if start not in self.directed_graph:
            print(f"在图中没有“{start}”")
            return None

        if end not in self.directed_graph:
            if print_result:
                print(f"在图中没有“{end}”")
            return None

        # Find all bridge words
        for neighbor in self.directed_graph[start]:
            if end in self.directed_graph.get(neighbor, {}):
                bridge_words.add(neighbor)
                if print_result:
                    print(f"桥接词为：{neighbor}")

        if not bridge_words and print_result:
            print(f"{start}和{end}之间没有桥接词")

        return bridge_words if bridge_words else None

    def find_all_paths(self, current: str, end: str, visited: Set[str], path: List[str], current_length: int) -> List[Tuple[List[str], int]]:
        '''
        每次从当前节点出发，尝试走向所有未访问过的邻居；
        每一步都更新路径 path 和当前路径长度；
        如果到达终点，就保存这条路径；
        递归返回所有可能的路径；
        回溯恢复状态，继续探索其他路径。
        '''
        # 如果当前节点已经到达终点
        if current == end:
            new_path = path.copy()        # 复制当前路径
            new_path.append(current)      # 将当前节点加入路径末尾
            return [(new_path, current_length)]  # 返回路径及其总长度

        visited.add(current)              # 将当前节点标记为已访问
        all_paths = []                    # 用于存储从当前节点到终点的所有路径

        # 遍历当前节点的所有邻居及其边的权重
        for neighbor, weight in self.directed_graph.get(current, {}).items():
            if neighbor not in visited:   # 避免重复访问节点（防止回路）
                path.append(current)
                # 递归查找从邻居到终点的所有路径
                neighbor_paths = self.find_all_paths(neighbor, end, visited, path, current_length + weight)
                all_paths.extend(neighbor_paths)  # 将找到的路径加入结果集中
                path.pop()                # 回溯：移除刚才加入的节点，准备探索其他路径

        visited.remove(current)          # 回溯：将当前节点从已访问集合中移除
        return all_paths                 # 返回所有路径和它们的长度


    def generate_new_text(self, input_text: str) -> str:
        words = input_text.split()
        new_text = []

        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            new_text.append(current_word)

            bridge_words = self.query_bridge_words(current_word, next_word)
            if bridge_words:
                bridge_word = random.choice(list(bridge_words))
                new_text.append(bridge_word)

        new_text.append(words[-1])
        result = " ".join(new_text)
        print(f"生成的新文本为: {result}")
        return result

    def calc_shortest_path(self, word1: str, word2: str, i: int, root: str) -> Optional[List[Tuple[List[str], int]]]:
        all_paths = self.find_all_paths(word1, word2, set(), [], 0)

        if not all_paths:
            return None

        # print(f"The path between {word1} {word2} is: ")
        # for path, length in all_paths:
        #     print(f"Path: {' -> '.join(path)}, Length: {length}")

        min_length = min(length for _, length in all_paths)
        shortest_paths = [(path, length) for path, length in all_paths if length == min_length]

        dot_file_path = f"./graph/directed_graph_shortest{i}.dot"
        self.create_dot_file(dot_file_path, shortest_paths, root)

        return shortest_paths

    def page_rank(self, text, d: float = 0.85, max_iterations: int = 100, tol: float = 1e-6, use_tfidf: bool = True) -> Dict[str, float]:
        """
        计算 PageRank 值
        d: 阻尼系数（一般设为 0.85）
        max_iterations: 最大迭代次数
        tol: 收敛容忍度
        use_tfidf: 是否使用 TF-IDF 值初始化 PR（若为 True，则基于词频初始化 PR 值）
        """
        print("开始计算 PageRank 值...")

        nodes = list(self.directed_graph.keys())  # 图中的所有节点
        N = len(nodes)  # 节点总数
        pr = dict.fromkeys(nodes, 1.0 / N)  # 初始 PR 值均匀分配

        if use_tfidf:
            print("使用 TF-IDF 初始化 PR 值...")
            # 1. 计算 TF（总词频） 该词与其他词的共现强度之和
            tf = {node: sum(edges.values()) for node, edges in self.directed_graph.items()}

            # 2. 计算 DF（出现在多少“句子”中），这里将文本按句子切分，每个句子视为一个“文档”
            sentence_list = text.lower().split('.')
            df = {node: 0 for node in nodes}
            for sentence in sentence_list:
                unique_words = set(sentence.strip().split())
                for word in unique_words:
                    if word in df:
                        df[word] += 1

            N_docs = len(sentence_list) + 1  # 防止除 0
            tfidf = {}
            for word in nodes:
                idf = math.log(N_docs / (df[word] + 1))
                tfidf[word] = tf[word] * idf

            # 归一化 TF-IDF 作为初始 PR
            total = sum(tfidf.values())
            pr = {word: tfidf[word] / total for word in tfidf}
        out_degree = {node: sum(edges.values()) for node, edges in self.directed_graph.items()}  # 计算每个节点的出度

        for iteration in range(max_iterations):
            new_pr = {}  # 存储每次迭代后的新 PR 值

            for node in nodes:
                new_value = (1 - d) / N  # 计算节点 PR 的初始值，（1-d）/N 是PR的泄漏部分

                # 更新 PR 值（根据入边权重计算）
                for other_node in nodes:
                    if node in self.directed_graph[other_node]:  # 如果 other_node 指向 node
                        # 这里的 L(u) 是指节点 u 的出度
                        out_degree_other_node = out_degree[other_node]  # 获取节点 other_node 的出度
                        # 根据 PageRank 公式更新新值
                        new_value += d * pr[
                            other_node] / out_degree_other_node  # 公式中的 d * (PR(other_node) / L(other_node))

                new_pr[node] = new_value  # 更新 node 的 PR 值

            # 处理出度为0的节点（leak），将它们的 PR 值均匀分配到所有节点
            leak_contrib = sum(pr[node] for node in nodes if out_degree.get(node, 0) == 0)  # 出度为 0 的节点的 PR 值总和
            leak_contrib *= d / N  # 将其均匀分配给所有节点，泄漏贡献
            for node in new_pr:
                new_pr[node] += leak_contrib  # 每个节点都增加来自泄漏节点的 PR 值

            # 检查是否收敛（当两次 PR 值差异小于容忍度时，认为收敛）
            delta = sum(abs(new_pr[node] - pr[node]) for node in nodes)  # 计算 PR 值的变化量
            pr = new_pr  # 更新 PR 值
            # print(f"迭代 {iteration+1}: 总变化量 {delta:.6f}")
            if delta < tol:  # 如果收敛则跳出循环
                break

        # 对 PR 值进行排序，按照 PR 值从高到低排序
        sorted_pr = dict(sorted(pr.items(), key=lambda x: x[1], reverse=True))
        # print("Top 10 PageRank 节点:")
        # for i, (node, value) in enumerate(list(sorted_pr.items())[:10]):
        #     print(f"{i+1}. {node} - PR值: {value:.6f}")
        #
        return sorted_pr

    def random_walk(self) -> str:
        vertices = list(self.directed_graph.keys())
        current_vertex = random.choice(vertices)

        visited_vertices = []
        visited_edges = []

        while not self.stop_walk.is_set():
            visited_vertices.append(current_vertex)
            edges = self.directed_graph.get(current_vertex, {})

            if edges:
                next_vertex = random.choice(list(edges.keys()))
                edge = f"{current_vertex} -> {next_vertex}"
                visited_edges.append(edge)
                current_vertex = next_vertex
                time.sleep(0.1)
            else:
                break

        output_path = "./random_walk.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("随机游走的节点:\n")
            f.write(" ".join(visited_vertices) + "\n")
            f.write("\n遍历的边:\n")
            f.write("\n".join(visited_edges) + "\n")

        print(f"随机游走的结果已写入文件：{output_path}")
        return output_path


def main():
    file_path = "./test/Easy Test.txt"
    dot_file_path = "./graph/directed_graph.dot"

    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    # 创建必要的目录
    os.makedirs("./test", exist_ok=True)
    os.makedirs("./graph", exist_ok=True)

    graph_builder = TextToGraph()
    all_words = graph_builder.show_directed_graph(file_path, dot_file_path, "")
    root_word = all_words[0]

    while True:
        print("请选择操作：")
        print("1. 桥接词查询")
        print("2. 生成新文本")
        print("3. 计算两个单词之间最短路径")
        print("4. 开始随机游走")
        print("5. 计算 PageRank")
        print("6. 退出")

        print("请输入操作编号：", end="")

        try:
            choice = int(input())
        except ValueError:
            print("请输入有效的数字！")
            continue

        if choice == 1:
            print("桥接词查询：")
            print("请输入两个词语（用空格分隔）：", end="")
            words = input().strip().split()
            if len(words) != 2:
                print("要求输入词语数量为2！")
                continue

            graph_builder.query_bridge_words(words[0].lower(), words[1].lower())

        elif choice == 2:
            print("根据bridge word生成新文本：")
            print("请输入文本：", end="")
            input_sentence = input().strip()
            graph_builder.generate_new_text(input_sentence)

        elif choice == 3:
            print("计算两个单词之间最短路径：")
            print("请输入一个或两个词语（用空格分隔）：", end="")
            words = input().strip().split()

            if len(words) == 1:
                for i, word in enumerate(all_words[1:], 1):
                    graph_builder.calc_shortest_path(words[0].lower(), word.lower(), i, root_word)
            elif len(words) == 2:
                graph_builder.calc_shortest_path(words[0].lower(), words[1].lower(), 1, root_word)
            else:
                print("要求输入词语数量为1或2！")

        elif choice == 4:
            print("开始随机游走")
            print("输入任意键以停止随机游走...")

            graph_builder.stop_walk.clear()
            walk_thread = Thread(target=graph_builder.random_walk)
            walk_thread.start()

            # Wait for user input to stop
            input()
            graph_builder.stop_walk.set()
            walk_thread.join()

        elif choice == 5:
            pr_scores = graph_builder.page_rank()
            print("是否要查询某个特定单词的 PageRank 值？(y/n): ", end="")
            answer = input().strip().lower()
            if answer == 'y':
                print("请输入要查询的单词：", end="")
                word = input().strip().lower()
                if word in pr_scores:
                    print(f"{word} 的 PageRank 值为: {pr_scores[word]:.6f}")
                else:
                    print(f"{word} 不在图中。")

        elif choice == 6:
            print("退出程序")
            break
        else:
            print("无效的操作编号，请重新输入！")


if __name__ == "__main__":
    main()
