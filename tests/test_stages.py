"""
test_structures: unit tests for workflow structures
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""

import dataclasses

import denovo

import fiat


@dataclasses.dataclass
class Something(denovo.structures.Node):
    
    pass


@dataclasses.dataclass
class AnotherThing(denovo.structures.Node):
    
    pass


@dataclasses.dataclass
class EvenAnother(denovo.structures.Node):
    
    pass


def test_workflow():
    # Tests adjacency matrix constructor
    matrix = tuple([[[0, 0, 1], [1, 0, 0], [0, 0, 0]],
                    ['scorpion', 'frog', 'river']])
    workflow = denovo.Workflow.from_matrix(matrix = matrix)
    assert 'scorpion' in workflow['frog']
    assert 'river' not in workflow['frog']
    # Tests adjacency list constructor
    adjacency = {'grumpy': {'sleepy'},
                 'doc': {},
                 'sneezy': {'grumpy', 'bashful'}}
    workflow = denovo.Workflow.from_adjacency(adjacency = adjacency)
    assert 'sleepy' in workflow['grumpy']
    assert 'bashful' in workflow['sneezy']
    assert 'bashful' not in workflow['doc']
    # Tests edge list constructor
    edges = [('camera', 'woman'), 
             ('camera', 'man'), 
             ('person', 'man'), 
             ('tv', 'person')]
    workflow_edges = denovo.Workflow.from_edges(edges = edges)
    assert 'woman' in workflow_edges['camera']
    assert 'man' in workflow_edges['camera']
    assert 'tv' not in workflow_edges['person']
    # Tests manual construction
    workflow = denovo.Workflow()
    workflow.add('bonnie')
    workflow.add('clyde')
    workflow.add('butch')
    workflow.add('sundance')
    workflow.add('henchman')
    workflow.connect('bonnie', 'clyde')
    workflow.connect('butch', 'sundance')
    workflow.connect('bonnie', 'henchman')
    workflow.connect('sundance', 'henchman')
    assert 'clyde' in workflow['bonnie']
    assert 'henchman' in workflow ['bonnie']
    assert 'henchman' not in workflow['butch']
    # Tests searches and paths
    # depth_search = workflow.search()
    # assert depth_search == ['bonnie', 'clyde', 'henchman']
    # breadth_search = workflow.search(depth_first = False)
    # print(breadth_search)
    # assert breadth_search == ['clyde', 'bonnie', 'henchman']
    all_paths = workflow.paths
    assert ['butch', 'sundance', 'henchman'] in all_paths
    assert ['bonnie', 'clyde'] in all_paths
    assert ['bonnie', 'henchman'] in all_paths
    workflow.merge(item = workflow_edges)
    new_workflow = denovo.Workflow()
    something = Something()
    another_thing = AnotherThing()
    even_another = EvenAnother()
    new_workflow.add(node = something)
    new_workflow.add(node = another_thing)
    new_workflow.add(node= even_another)
    new_workflow.connect('something', 'another_thing')
    assert 'another_thing' in new_workflow['something']
    assert 'another_thing' in new_workflow[something]
    assert another_thing in new_workflow[something]
    assert something in new_workflow
    return


if __name__ == '__main__':
    denovo.testing.testify(target_module = fiat.structures, 
                           testing_module = __name__)
    
