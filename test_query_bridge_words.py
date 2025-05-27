import pytest
from lab1 import TextToGraph

@pytest.fixture
def graph():
    tg = TextToGraph()
    file_path = "./test/Easy Test.txt"
    dot_file_path = "./graph/directed_graph.dot"
    # 手动构造有向图
 
    tg.show_directed_graph(file_path, dot_file_path, "")
    return tg

def test_case_1(graph, capsys):
    # ("the", "carefully")，有桥接词
    result = graph.query_bridge_words("the", "carefully")
    captured = capsys.readouterr().out
    assert result == {"scientist"}
    assert "桥接词为：scientist" in captured

def test_case_2(graph, capsys):
    # ("banana", "data")，banana 不在图中
    result = graph.query_bridge_words("banana", "data")
    captured = capsys.readouterr().out
    assert result is None
    assert "在图中没有“banana”" in captured

def test_case_3(graph, capsys):
    # ("the", "orange")，orange 不在图中
    result = graph.query_bridge_words("the", "orange")
    captured = capsys.readouterr().out
    assert result is None
    assert "在图中没有“orange”" in captured

def test_case_4(graph, capsys):
    # ("apple", "orange")，两者都不在图中（但都在节点里，只是没有边）
    result = graph.query_bridge_words("apple", "orange")
    captured = capsys.readouterr().out
    assert result is None
    # 这里可以根据实际实现调整断言

def test_case_5(graph, capsys):
    # ("the", "wrote")，两者不在连接词
    result = graph.query_bridge_words("the", "a")
    captured = capsys.readouterr().out
    assert result is None
    assert "the和a之间没有桥接词" in captured

def test_case_6(graph, capsys):
    # ("@", "#so")，非法字符
    result = graph.query_bridge_words("@", "#so")
    captured = capsys.readouterr().out
    assert result is None
    assert "@” 不是合法的小写单词" in captured

def test_case_7(graph, capsys):
    # ("the", "")，字符数小于2
    result = graph.query_bridge_words("the", "")
    captured = capsys.readouterr().out
    assert result is None
    assert "单词不能为空" in captured

def test_case_8(graph):
    # ("the", "so", "data")，参数数量不符
    with pytest.raises(TypeError):
        graph.query_bridge_words("the", "so", "data")