import pytest
from lab1 import TextToGraph
import os

@pytest.fixture
def graph():
    """创建测试用的图实例"""
    graph = TextToGraph()
    test_file = os.path.join("test", "test_shortest_path.txt")
    graph.build_directed_graph(test_file)
    return graph

def test_no_path(graph):
    """测试用例1：两个节点之间没有路径"""
    result = graph.calc_shortest_path("again", "the", 1, "with")
    assert result is None

def test_path_with_multiple_steps(graph):
    """测试用例2：两个节点之间有多个步骤的路径"""
    result = graph.calc_shortest_path("with", "wrote", 2, "with")
    assert result is not None
    assert len(result) == 1
    path, length = result[0]
    assert path == ['with', 'the', 'data', 'wrote']
    assert length == 3

def test_direct_path(graph):
    """测试用例3：两个节点之间有直接路径"""
    result = graph.calc_shortest_path("the", "data", 3, "with")
    assert result is not None
    assert len(result) == 1
    path, length = result[0]
    assert path == ['the', 'data']
    assert length == 1

def test_path_with_two_steps(graph):
    """测试用例4：两个节点之间有两个步骤的路径"""
    result = graph.calc_shortest_path("analyzed", "again", 4, "with")
    assert result is not None
    assert len(result) == 1
    path, length = result[0]
    assert path == ['analyzed', 'it', 'again']
    assert length == 2 