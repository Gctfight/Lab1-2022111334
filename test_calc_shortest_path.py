import pytest
from lab1 import TextToGraph
import os

@pytest.fixture
def graph():
    tg = TextToGraph()
    file_path = "./test/Easy Test.txt"
    dot_file_path = "./graph/directed_graph.dot"
    # 手动构造有向图
 
    tg.show_directed_graph(file_path, dot_file_path, "")
    return tg

def test_case_1(graph):
    """测试用例1：两个节点之间没有路径"""
    result = graph.calc_shortest_path("again", "the", 1, "the")
    assert result is None

def test_case_2(graph):
    """测试用例2：两个节点之间有多个步骤的路径"""
    result = graph.calc_shortest_path("with", "wrote", 2, "the")
    assert result is not None
    assert len(result) == 1
    path, length = result[0]
    assert path == ['with', 'the', 'data', 'wrote']
    assert length == 3

def test_case_3(graph):
    """测试用例3：两个节点之间有直接路径"""
    result = graph.calc_shortest_path("the", "data", 3, "the")
    assert result is not None
    assert len(result) == 1
    path, length = result[0]
    assert path == ['the', 'data']
    assert length == 1

def test_case_4(graph):
    """测试用例4：两个节点之间有两个步骤的路径"""
    result = graph.calc_shortest_path("analyzed", "again", 4, "the")
    assert result is not None
    assert len(result) == 1
    path, length = result[0]
    assert path == ['analyzed', 'it', 'again']
    assert length == 2