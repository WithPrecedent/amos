"""
test_core: unit tests for amos data structures
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""
import dataclasses

import amos


@dataclasses.dataclass
class Something(amos.Node):
    
    pass


@dataclasses.dataclass
class AnotherThing(amos.Node):
    
    pass


@dataclasses.dataclass
class EvenAnother(amos.Node):
    
    pass


def test_graph() -> None:
    edges = tuple([('a', 'b'), ('c', 'd'), ('a', 'd'), ('d', 'e')])
    dag = amos.System.create(item = edges)
    dag.add(node = 'cat')
    dag.add(node = 'dog', ancestors = 'e', descendants = ['cat'])
    assert dag['dog'] == {'cat'}
    assert dag['e'] == {'dog'}
    adjacency = {
        'tree': {'house', 'yard'},
        'house': set(),
        'yard': set()}
    assert amos.Adjacency.__instancecheck__(adjacency)
    another_dag = amos.System.create(item = adjacency)
    dag.append(item = another_dag)
    assert dag['cat'] == {'tree'}
    pipelines = dag.pipelines 
    assert len(pipelines) == 6
    assert dag.endpoint == {'house', 'yard'}
    assert dag.root == {'a', 'c'}
    assert dag.nodes == {
        'tree', 'b', 'c', 'a', 'yard', 'cat', 'd', 'house', 'dog', 'e'}
    pipeline = dag.pipeline
    new_dag = amos.System.from_pipeline(item = pipeline)
    assert new_dag['tree'] == dag['tree']
    another_dag = amos.System.from_pipelines(item = pipelines)
    assert another_dag['tree'] == dag['tree']
    return

def test_pipeline() -> None:
    
    return

def test_tree() -> None:
    
    return


if __name__ == '__main__':
    test_graph()
    test_pipeline()
    test_tree()
    