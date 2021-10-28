"""
graph: lightweight graph data structures
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2021, Corey Rayburn Yung
License: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Classes:
    System (Graph): a lightweight directed acyclic graph (DAG). Internally, the 
        graph is stored as an adjacency list. As a result, it should primarily 
        be used for workflows or other uses that do require large graphs.
        
To Do:
    Complete Network which will use an adjacency matrix for internal storage.
    
"""
from __future__ import annotations
import abc
import collections
from collections.abc import (
    Collection, Hashable, MutableMapping, MutableSequence, Sequence, Set)
import copy
import dataclasses
import itertools
from typing import Any, Optional, Type, TYPE_CHECKING, Union

from ..base import mappings
from ..observe import traits
from ..repair import convert
from . import composites
from . import check
from . import transform
from . import hybrid

if TYPE_CHECKING:
    from . import tree
    
 
@dataclasses.dataclass
class Edge(composites.Composite, abc.ABC):
    
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_edge(item = instance)

     
@dataclasses.dataclass # type: ignore
class Graph(composites.Composite, abc.ABC):
    """composites class for graph data structures.
    
    Args:
        contents (Collection[Any]): stored collection of nodes and/or edges.
                                      
    """  
    contents: Collection[Any]

    """ Required Subclass Properties """

    @abc.abstractmethod
    def connect(self, start: composites.Node, stop: composites.Node) -> None:
        """Creates a new edge in the stored graph.

        Args:
            start (Node): starting node for the new edge.
            stop (Node): ending node for the new edge.
        
        Raises:
            KeyError: if 'start' or 'stop' are not a nodes in the stored graph.

        """
        pass  
    
    @abc.abstractmethod
    def disconnect(self, start: composites.Node, stop: composites.Node) -> None:
        """Deletes edge from the stored graph.

        Args:
            start (Node): starting node for the edge to delete.
            stop (Node): ending node for the edge to delete.
        
        Raises:
            KeyError: if 'start' or 'stop' are not a nodes in the stored graph.

        """
        pass  
    
    """ Dunder Methods """
    
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_graph(item = instance)


@dataclasses.dataclass
class Adjacency(mappings.Dictionary, Graph):
    """composites class for adjacency-list-compositesd graphs.
    
    Args:
        contents (MutableMapping[composites.Node, Set[composites.Node]]): keys are nodes and 
            values are sets of nodes (or hashable representations of nodes). 
            Defaults to a defaultdict that has a set for its value format.
                                      
    """  
    contents: MutableMapping[composites.Node, Set[composites.Node]] = dataclasses.field(
        default_factory = lambda: collections.defaultdict(set))
    
    """ Dunder Methods """
    
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_adjacency(item = instance)


@dataclasses.dataclass
class Edges(Graph):
    """composites class for edges-list-compositesd graphs.
    
    Args:
        contents (tuple[tuple[composites.Node, Node], ...]): tuple of tuple of edges. 
            Defaults to an empty tuple.
                                      
    """   
    contents: tuple[tuple[composites.Node, composites.Node], ...] = dataclasses.field(
        default_factory = tuple)
    
    """ Dunder Methods """
           
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_edges(item = instance)
    
    
@dataclasses.dataclass
class Matrix(Graph):
    """composites class for adjacency-matrix-compositesd graphs.
    
    Args:
        contents (Sequence[Sequence[int]]): a list of list of integers 
            indicating edges between nodes in the matrix. Defaults to an empty
            list.
        labels (Sequence[composites.Node]): names of nodes in the matrix. Defaults to 
            an empty list.
                                      
    """  
    contents: Sequence[Sequence[int]] = dataclasses.field(
        default_factory = list)
    labels: Sequence[composites.Node] = dataclasses.field(default_factory = list)
    
    """ Dunder Methods """
        
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_matrix(item = instance)

    
@dataclasses.dataclass
class System(Adjacency):
    """Directed graph with unweighted edges.
    
    Args:
        contents (MutableMapping[composites.Node, Set[composites.Node]]): keys are nodes and 
            values are sets of nodes (or hashable representations of nodes). 
            Defaults to a defaultdict that has a set for its value format.
                  
    """  
    contents: MutableMapping[composites.Node, Set[composites.Node]] = dataclasses.field(
        default_factory = lambda: collections.defaultdict(set))
    
    """ Properties """

    @property
    def endpoint(self) -> set[composites.Node]:
        """Returns endpoint nodes in the stored graph in a list."""
        return {k for k in self.contents.keys() if not self.contents[k]}
                    
    @property
    def root(self) -> set[composites.Node]:
        """Returns root nodes in the stored graph in a list."""
        stops = list(itertools.chain.from_iterable(self.contents.values()))
        return {k for k in self.contents.keys() if k not in stops}

    @property
    def adjacency(self) -> Adjacency:
        """Returns the stored graph as an adjacency list."""
        return self.contents

    @property
    def edges(self) -> Edges:
        """Returns the stored graph as an edge list."""
        return transform.adjacency_to_edges(item = self.contents)
    
    @property
    def matrix(self) -> Matrix:
        """Returns the stored graph as an adjacency matrix."""
        return transform.adjacency_to_matrix(item = self.contents)
                      
    @property
    def nodes(self) -> set[composites.Node]:
        """Returns all stored nodes in a list."""
        return set(self.contents.keys())

    @property
    def paths(self) -> composites.Nodes:
        """Returns all paths through the stored as a list of nodes."""
        return self._find_all_paths(starts = self.root, stops = self.endpoint)
    
    @property
    def pipeline(self) -> hybrid.Pipeline:
        """Returns stored graph as a pipeline."""
        pipeline = []
        for pipe in self.pipelines.values():
            pipeline.extend(pipe)
        return hybrid.Pipeline(contents = pipeline)
    
    @property
    def pipelines(self) -> hybrid.Pipelines:
        """Returns stored graph as pipelines."""
        all_paths = self.paths
        instances = [hybrid.Pipeline(contents = p) for p in all_paths]
        pipelines = hybrid.Pipelines()
        for instance in instances:
            pipelines.add(instance, name = 'path')
        return pipelines
            
    @property
    def tree(self) -> tree.Tree:
        """Returns the stored composite object as a tree.Tree."""
        raise NotImplementedError

    """ Class Methods """
 
    @classmethod
    def from_adjacency(cls, item: Adjacency) -> System:
        """Creates a System instance from an adjacency list."""
        return cls(contents = item)
    
    @classmethod
    def from_edges(cls, item: Edges) -> System:
        """Creates a System instance from an edge list."""
        return cls(contents = transform.edges_to_adjacency(item = item))
    
    @classmethod
    def from_matrix(cls, item: Matrix) -> System:
        """Creates a System instance from an adjacency matrix."""
        return cls(contents = transform.matrix_to_adjacency(item = item))
    
    @classmethod
    def from_nodes(cls, item: composites.Nodes) -> System:
        """Creates a System instance from a Nodes."""
        new_contents = transform.pipeline_to_adjacency(item = item)
        return cls(contents = new_contents)

    @classmethod
    def from_pipeline(cls, item: hybrid.Pipeline) -> System:
        """Creates a System instance from a Pipeline."""
        new_contents = transform.pipeline_to_adjacency(item = item)
        return cls(contents = new_contents)
    
    @classmethod
    def from_pipelines(cls, item: hybrid.Pipelines) -> System:
        """Creates a System instance from a Pipeline."""
        new_contents = transform.pipelines_to_adjacency(item = item)
        return cls(contents = new_contents)

    @classmethod
    def from_tree(cls, item: tree.Tree) -> System:
        """Creates a System instance from a Tree."""
        raise NotImplementedError
             
    """ Public Methods """

    def add(
        self, 
        node: composites.Node,
        ancestors: composites.Nodes = None,
        descendants: composites.Nodes = None) -> None:
        """Adds 'node' to the stored graph.
        
        Args:
            node (composites.Node): a node to add to the stored graph.
            ancestors (composites.Nodes): node(s) from which 'node' should be 
                connected.
            descendants (composites.Nodes): node(s) to which 'node' should be 
                connected.

        Raises:
            KeyError: if some nodes in 'descendants' or 'ancestors' are not in 
                the stored graph.
                
        """
        if descendants is None:
            self.contents[node] = set()
        # elif utilities.is_property(item = descendants, instance = self):
        #     self.contents = set(getattr(self, descendants))
        else:
            descendants = list(convert.iterify(item = descendants))
            descendants = [traits.get_name(item = n) for n in descendants]
            missing = [n for n in descendants if n not in self.contents]
            if missing:
                raise KeyError(
                    f'descendants {str(missing)} are not in '
                    f'{self.__class__.__name__}')
            else:
                self.contents[node] = set(descendants)
        if ancestors is not None:  
            # if utilities.is_property(item = ancestors, instance = self):
            #     start = list(getattr(self, ancestors))
            # else:
            ancestors = list(convert.iterify(item = ancestors))
            missing = [n for n in ancestors if n not in self.contents]
            if missing:
                raise KeyError(
                    f'ancestors {str(missing)} are not in '
                    f'{self.__class__.__name__}')
            for start in ancestors:
                if node not in self[start]:
                    self.connect(start = start, stop = node)                 
        return 

    def append(self, item: composites.Composite) -> None:
        """Appends 'item' to the endpoints of the stored graph.

        Appending creates an edge between every endpoint of this instance's
        stored graph and the every root of 'item'.

        Args:
            item (composites.Composite): another Graph, 
                an adjacency list, an edge list, an adjacency matrix, or one or
                more nodes.
            
        Raises:
            TypeError: if 'item' is neither a Graph, Adjacency, Edges, Matrix,
                or composites.Nodes type.
                
        """
        if isinstance(item, composites.Composite):
            current_endpoints = list(self.endpoint)
            new_graph = self.create(item = item)
            self.merge(item = new_graph)
            for endpoint in current_endpoints:
                for root in new_graph.root:
                    self.connect(start = endpoint, stop = root)
        else:
            raise TypeError('item must be a Node, Nodes, or Composite type')
        return
  
    def connect(self, start: composites.Node, stop: composites.Node) -> None:
        """Adds an edge from 'start' to 'stop'.

        Args:
            start (composites.Node): name of node for edge to start.
            stop (composites.Node): name of node for edge to stop.
            
        Raises:
            ValueError: if 'start' is the same as 'stop'.
            
        """
        if start == stop:
            raise ValueError(
                'The start of an edge cannot be the same as the '
                'stop in a System because it is acyclic')
        elif start not in self:
            self.add(node = start)
        elif stop not in self:
            self.add(node = stop)
        if stop not in self.contents[start]:
            self.contents[start].add(traits.get_name(item = stop))
        return

    def delete(self, node: composites.Node) -> None:
        """Deletes node from graph.
        
        Args:
            node (composites.Node): node to delete from 'contents'.
        
        Raises:
            KeyError: if 'node' is not in 'contents'.
            
        """
        try:
            del self.contents[node]
        except KeyError:
            raise KeyError(f'{node} does not exist in the graph')
        self.contents = {k: v.discard(node) for k, v in self.contents.items()}
        return

    def disconnect(self, start: composites.Node, stop: composites.Node) -> None:
        """Deletes edge from graph.

        Args:
            start (composites.Node): starting node for the edge to delete.
            stop (composites.Node): ending node for the edge to delete.
        
        Raises:
            KeyError: if 'start' is not a node in the stored graph..

        """
        try:
            self.contents[start].discard(stop)
        except KeyError:
            raise KeyError(f'{start} does not exist in the graph')
        return

    def merge(self, item: composites.Composite) -> None:
        """Adds 'item' to this Graph.

        This method is roughly equivalent to a dict.update, just adding the
        new keys and values to the existing graph. It transforms 'item' to an 
        adjacency list that is then added to the existing 'contents'.
        
        Args:
            item (composites.Composite): another Graph, an adjacency 
                list, an edge list, an adjacency matrix, or one or more nodes.
            
        Raises:
            TypeError: if 'item' is neither a System, Adjacency, 
                Edges, Matrix, or composites.Nodes type.
            
        """
        if isinstance(item, System):
            adjacency = item.adjacency
        elif isinstance(item, Adjacency):
            adjacency = item
        elif isinstance(item, Edges):
            adjacency = transform.edges_to_adjacency(item = item)
        elif isinstance(item, Matrix):
            adjacency = transform.matrix_to_adjacency(item = item)
        elif isinstance(item, (list, tuple, set)):
            adjacency = transform.pipeline_to_adjacency(item = item)
        elif isinstance(item, composites.Node):
            adjacency = {item: set()}
        else:
            raise TypeError('item must be a Node, Nodes, or Composite type')
        self.contents.update(adjacency)
        return

    def prepend(self, item: composites.Composite) -> None:
        """Prepends 'item' to the roots of the stored graph.

        Prepending creates an edge between every endpoint of 'item' and every
        root of this instance;s stored graph.

        Args:
            item (composites.Composite): another Graph, an adjacency list, an 
                edge list, an adjacency matrix, or one or more nodes.
            
        Raises:
            TypeError: if 'item' is neither a System, Adjacency, Edges, Matrix, 
                or composites.Nodes type.
                
        """
        if isinstance(item, composites.Composite):
            current_roots = list(self.root)
            new_graph = self.create(item = item)
            self.merge(item = new_graph)
            for root in current_roots:
                for endpoint in new_graph.endpoints:
                    self.connect(start = endpoint, stop = root)
        else:
            raise TypeError(
                'item must be a System, Adjacency, Edges, Matrix, hybrid.Pipeline, '
                'hybrid.Pipelines, or Node type')
        return
      
    def subset(
        self, 
        include: Union[Any, Sequence[Any]] = None,
        exclude: Union[Any, Sequence[Any]] = None) -> System:
        """Returns a new System without a subset of 'contents'.
        
        All edges will be removed that include any nodes that are not part of
        the new subgraph.
        
        Any extra attributes that are part of a System (or a subclass) will be
        maintained in the returned subgraph.

        Args:
            include (Union[Any, Sequence[Any]]): nodes which should be included
                with any applicable edges in the new subgraph.
            exclude (Union[Any, Sequence[Any]]): nodes which should not be 
                included with any applicable edges in the new subgraph.

        Returns:
           System: with only key/value pairs with keys not in 'subset'.

        """
        if include is None and exclude is None:
            raise ValueError('Either include or exclude must not be None')
        else:
            if include:
                excludables = [k for k in self.contents if k not in include]
            else:
                excludables = []
            excludables.extend([i for i in self.contents if i in exclude])
            new_graph = copy.deepcopy(self)
            for node in convert.iterify(item = excludables):
                new_graph.delete(node = node)
        return new_graph
    
    def walk(
        self, 
        start: composites.Node,
        stop: composites.Node, 
        path: Optional[hybrid.Pipeline] = None) -> hybrid.Pipeline:
        """Returns all paths in graph from 'start' to 'stop'.

        The code here is adapted from: https://www.python.org/doc/essays/graphs/
        
        Args:
            start (composites.Node): node to start paths from.
            stop (composites.Node): node to stop paths.
            path (hybrid.Pipeline): a path from 'start' to 'stop'. Defaults 
                to an empty list. 

        Returns:
            hybrid.Pipeline: a list of possible paths (each path is a list 
                nodes) from 'start' to 'stop'.
            
        """
        if path is None:
            path = []
        path = path + [start]
        if start == stop:
            return [path]
        if start not in self.contents:
            return []
        paths = []
        for node in self.contents[start]:
            if node not in path:
                new_paths = self.walk(
                    start = node, 
                    stop = stop, 
                    path = path)
                for new_path in new_paths:
                    paths.append(new_path)
        return paths

    """ Private Methods """

    def _find_all_paths(
        self, 
        starts: composites.Nodes, 
        stops: composites.Nodes) -> hybrid.Pipeline:
        """Returns all paths between 'starts' and 'stops'.

        Args:
            start (composites.Nodes): starting point(s) for paths through the 
                System.
            ends (composites.Nodes): ending point(s) for paths through the 
                System.

        Returns:
            hybrid.Pipeline: list of all paths through the System from all 
                'starts' to all 'ends'.
            
        """
        all_paths = []
        for start in convert.iterify(item = starts):
            for end in convert.iterify(item = stops):
                paths = self.walk(start = start, stop = end)
                if paths:
                    if all(isinstance(path, composites.Node) for path in paths):
                        all_paths.append(paths)
                    else:
                        all_paths.extend(paths)
        return all_paths


# @dataclasses.dataclass
# class Network(Graph):
#     """composites class for undirected graphs with unweighted edges.
    
#     Graph stores a directed acyclic graph (DAG) as an adjacency list. Despite 
#     being called an adjacency "list," the typical and most efficient way to 
#     store one is using a python dict. a piles Graph inherits from a Dictionary 
#     in order to allow use of its extra functionality over a plain dict.
    
#     Graph supports '+' and '+=' to be used to join two piles Graph instances. A
#     properly formatted adjacency list could also be the added object.
    
#     Graph internally supports autovivification where a list is created as a 
#     value for a missing key. This means that a Graph need not inherit from 
#     defaultdict.
    
#     Args:
#         contents (Adjacency): an adjacency list where the keys are nodes and the 
#             values are nodes which the key is connected to. Defaults to an empty 
#             dict.
                  
#     """  
#     contents: Matrix = dataclasses.field(default_factory = dict)
    
#     """ Properties """

#     @property
#     def adjacency(self) -> Adjacency:
#         """Returns the stored graph as an adjacency list."""
#         return matrix_to_adjacency(item = self.contents)

#     @property
#     def breadths(self) -> hybrid.Pipeline:
#         """Returns all paths through the Graph using breadth-first search.
        
#         Returns:
#             hybrid.Pipeline: returns all paths from 'roots' to 'endpoints' in a list 
#                 of lists of nodes.
                
#         """
#         return self._find_all_paths(
#             starts = self.root, 
#             ends = self.endpoint,
#             depth_first = False)

#     @property
#     def depths(self) -> hybrid.Pipeline:
#         """Returns all paths through the Graph using depth-first search.
        
#         Returns:
#             hybrid.Pipeline: returns all paths from 'roots' to 'endpoints' in a list 
#                 of lists of nodes.
                
#         """
#         return self._find_all_paths(starts = self.root, 
#                                     ends = self.endpoint,
#                                     depth_first = True)
     
#     @property
#     def edges(self) -> Edges:
#         """Returns the stored graph as an edge list."""
#         return adjacency_to_edges(item = self.contents)

#     @property
#     def endpoints(self) -> list[composites.Node]:
#         """Returns a list of endpoint nodes in the stored graph.."""
#         return [k for k in self.contents.keys() if not self.contents[k]]

#     @property
#     def matrix(self) -> Matrix:
#         """Returns the stored graph as an adjacency matrix."""
#         return adjacency_to_matrix(item = self.contents)
                      
#     @property
#     def nodes(self) -> dict[str, composites.Node]:
#         """Returns a dict of node names as keys and nodes as values.
        
#         Because Graph allows various composites.Node objects to be used as keys,
#         including the composites.Nodes class, there isn't an obvious way to access already
#         stored nodes. This property creates a new dict with str keys derived
#         from the nodes (looking first for a 'name' attribute) so that a user
#         can access a node. 
        
#         This property is not needed if the stored nodes are all strings.
        
#         Returns:
#             Dict[str, composites.Node]: keys are the name or has of nodes and the 
#                 values are the nodes themselves.
            
#         """
#         return {self.traits.get_name(item = n): n for n in self.contents.keys()}
  
#     @property
#     def roots(self) -> list[composites.Node]:
#         """Returns root nodes in the stored graph..

#         Returns:
#             list[composites.Node]: root nodes.
            
#         """
#         stops = list(itertools.chain.from_iterable(self.contents.values()))
#         return [k for k in self.contents.keys() if k not in stops]
    
#     """ Class Methods """
    
#     @classmethod
#     def create(cls, item: Union[Adjacency, Edges, Matrix]) -> Graph:
#         """Creates an instance of a Graph from 'item'.
        
#         Args:
#             item (Union[Adjacency, Edges, Matrix]): an adjacency list, 
#                 adjacency matrix, or edge list which can used to create the
#                 stored graph.
                
#         Returns:
#             Graph: a Graph instance created compositesd on 'item'.
                
#         """
#         if is_adjacency_list(item = item):
#             return cls.from_adjacency(adjacency = item)
#         elif is_adjacency_matrix(item = item):
#             return cls.from_matrix(matrix = item)
#         elif is_edge_list(item = item):
#             return cls.from_adjacency(edges = item)
#         else:
#             raise TypeError(
#                 f'create requires item to be an adjacency list, adjacency '
#                 f'matrix, or edge list')
           
#     @classmethod
#     def from_adjacency(cls, adjacency: Adjacency) -> Graph:
#         """Creates a Graph instance from an adjacency list.
        
#         'adjacency' should be formatted with nodes as keys and values as lists
#         of names of nodes to which the node in the key is connected.

#         Args:
#             adjacency (Adjacency): adjacency list used to 
#                 create a Graph instance.

#         Returns:
#             Graph: a Graph instance created compositesd on 'adjacent'.
              
#         """
#         return cls(contents = adjacency)
    
#     @classmethod
#     def from_edges(cls, edges: Edges) -> Graph:
#         """Creates a Graph instance from an edge list.

#         'edges' should be a list of tuples, where the first item in the tuple
#         is the node and the second item is the node (or name of node) to which
#         the first item is connected.
        
#         Args:
#             edges (Edges): Edge list used to create a Graph 
#                 instance.
                
#         Returns:
#             Graph: a Graph instance created compositesd on 'edges'.

#         """
#         return cls(contents = edges_to_adjacency(item = edges))
    
#     @classmethod
#     def from_matrix(cls, matrix: Matrix) -> Graph:
#         """Creates a Graph instance from an adjacency matrix.

#         Args:
#             matrix (Matrix): adjacency matrix used to create a Graph instance. 
#                 The values in the matrix should be 1 (indicating an edge) and 0 
#                 (indicating no edge).
 
#         Returns:
#             Graph: a Graph instance created compositesd on 'matrix'.
                        
#         """
#         return cls(contents = matrix_to_adjacency(item = matrix))
    
#     @classmethod
#     def from_pipeline(cls, pipeline: hybrid.Pipeline) -> Graph:
#         """Creates a Graph instance from a Pipeline.

#         Args:
#             pipeline (hybrid.Pipeline): serial pipeline used to create a Graph
#                 instance.
 
#         Returns:
#             Graph: a Graph instance created compositesd on 'pipeline'.
                        
#         """
#         return cls(contents = pipeline_to_adjacency(item = pipeline))
       
#     """ Public Methods """
    
#     def add(self, 
#             node: composites.Node,
#             ancestors: composites.Nodes = None,
#             descendants: composites.Nodes = None) -> None:
#         """Adds 'node' to 'contents' with no corresponding edges.
        
#         Args:
#             node (composites.Node): a node to add to the stored graph.
#             ancestors (composites.Nodes): node(s) from which node should be connected.
#             descendants (composites.Nodes): node(s) to which node should be connected.

#         """
#         if descendants is None:
#             self.contents[node] = []
#         elif descendants in self:
#             self.contents[node] = convert.iterify(item = descendants)
#         else:
#             missing = [n for n in descendants if n not in self.contents]
#             raise KeyError(f'descendants {missing} are not in the stored graph.')
#         if ancestors is not None:  
#             if (isinstance(ancestors, composites.Node) and ancestors in self
#                     or (isinstance(ancestors, (list, tuple, set)) 
#                         and all(isinstance(n, composites.Node) for n in ancestors)
#                         and all(n in self.contents for n in ancestors))):
#                 start = ancestors
#             elif (hasattr(self.__class__, ancestors) 
#                     and isinstance(getattr(type(self), ancestors), property)):
#                 start = getattr(self, ancestors)
#             else:
#                 missing = [n for n in ancestors if n not in self.contents]
#                 raise KeyError(f'ancestors {missing} are not in the stored graph.')
#             for starting in convert.iterify(item = start):
#                 if node not in [starting]:
#                     self.connect(start = starting, stop = node)                 
#         return 

#     def append(self, 
#                item: Union[Graph, Adjacency, Edges, Matrix, composites.Nodes]) -> None:
#         """Adds 'item' to this Graph.

#         Combining creates an edge between every endpoint of this instance's
#         Graph and the every root of 'item'.

#         Args:
#             item (Union[Graph, Adjacency, Edges, Matrix, composites.Nodes]): another 
#                 Graph to join with this one, an adjacency list, an edge list, an
#                 adjacency matrix, or composites.Nodes.
            
#         Raises:
#             TypeError: if 'item' is neither a Graph, Adjacency, Edges, Matrix,
#                 or composites.Nodes type.
            
#         """
#         if isinstance(item, Graph):
#             if self.contents:
#                 current_endpoints = self.endpoint
#                 self.contents.update(item.contents)
#                 for endpoint in current_endpoints:
#                     for root in item.root:
#                         self.connect(start = endpoint, stop = root)
#             else:
#                 self.contents = item.contents
#         elif isinstance(item, Adjacency):
#             self.append(item = self.from_adjacency(adjacecny = item))
#         elif isinstance(item, Edges):
#             self.append(item = self.from_edges(edges = item))
#         elif isinstance(item, Matrix):
#             self.append(item = self.from_matrix(matrix = item))
#         elif isinstance(item, composites.Nodes):
#             if isinstance(item, (list, tuple, set)):
#                 new_graph = Graph()
#                 edges = more_itertools.windowed(item, 2)
#                 for edge_pair in edges:
#                     new_graph.add(node = edge_pair[0], descendants = edge_pair[1])
#                 self.append(item = new_graph)
#             else:
#                 self.add(node = item)
#         else:
#             raise TypeError(
#                 'item must be a Graph, Adjacency, Edges, Matrix, or composites.Nodes '
#                 'type')
#         return
  
#     def connect(self, start: composites.Node, stop: composites.Node) -> None:
#         """Adds an edge from 'start' to 'stop'.

#         Args:
#             start (composites.Node): name of node for edge to start.
#             stop (composites.Node): name of node for edge to stop.
            
#         Raises:
#             ValueError: if 'start' is the same as 'stop'.
            
#         """
#         if start == stop:
#             raise ValueError(
#                 'The start of an edge cannot be the same as the stop')
#         else:
#             if stop not in self.contents:
#                 self.add(node = stop)
#             if start not in self.contents:
#                 self.add(node = start)
#             if stop not in self.contents[start]:
#                 self.contents[start].append(self.traits.get_name(item = stop))
#         return

#     def delete(self, node: composites.Node) -> None:
#         """Deletes node from graph.
        
#         Args:
#             node (composites.Node): node to delete from 'contents'.
        
#         Raises:
#             KeyError: if 'node' is not in 'contents'.
            
#         """
#         try:
#             del self.contents[node]
#         except KeyError:
#             raise KeyError(f'{node} does not exist in the graph')
#         self.contents = {
#             k: v.remove(node) for k, v in self.contents.items() if node in v}
#         return

#     def disconnect(self, start: composites.Node, stop: composites.Node) -> None:
#         """Deletes edge from graph.

#         Args:
#             start (composites.Node): starting node for the edge to delete.
#             stop (composites.Node): ending node for the edge to delete.
        
#         Raises:
#             KeyError: if 'start' is not a node in the stored graph..
#             ValueError: if 'stop' does not have an edge with 'start'.

#         """
#         try:
#             self.contents[start].remove(stop)
#         except KeyError:
#             raise KeyError(f'{start} does not exist in the graph')
#         except ValueError:
#             raise ValueError(f'{stop} is not connected to {start}')
#         return

#     def merge(self, item: Union[Graph, Adjacency, Edges, Matrix]) -> None:
#         """Adds 'item' to this Graph.

#         This method is roughly equivalent to a dict.update, just adding the
#         new keys and values to the existing graph. It transforms the supported
#         formats to an adjacency list that is then added to the existing 
#         'contents'.
        
#         Args:
#             item (Union[Graph, Adjacency, Edges, Matrix]): another Graph to 
#                 add to this one, an adjacency list, an edge list, or an
#                 adjacency matrix.
            
#         Raises:
#             TypeError: if 'item' is neither a Graph, Adjacency, Edges, or 
#                 Matrix type.
            
#         """
#         if isinstance(item, Graph):
#             item = item.contents
#         elif isinstance(item, Adjacency):
#             pass
#         elif isinstance(item, Edges):
#             item = self.from_edges(edges = item).contents
#         elif isinstance(item, Matrix):
#             item = self.from_matrix(matrix = item).contents
#         else:
#             raise TypeError(
#                 'item must be a Graph, Adjacency, Edges, or Matrix type to '
#                 'update')
#         self.contents.update(item)
#         return
  
#     def subgraph(self, 
#                  include: Union[Any, Sequence[Any]] = None,
#                  exclude: Union[Any, Sequence[Any]] = None) -> Graph:
#         """Returns a new Graph without a subset of 'contents'.
        
#         All edges will be removed that include any nodes that are not part of
#         the new subgraph.
        
#         Any extra attributes that are part of a Graph (or a subclass) will be
#         maintained in the returned subgraph.

#         Args:
#             include (Union[Any, Sequence[Any]]): nodes which should be included
#                 with any applicable edges in the new subgraph.
#             exclude (Union[Any, Sequence[Any]]): nodes which should not be 
#                 included with any applicable edges in the new subgraph.

#         Returns:
#             Graph: with only key/value pairs with keys not in 'subset'.

#         """
#         if include is None and exclude is None:
#             raise ValueError('Either include or exclude must not be None')
#         else:
#             if include:
#                 excludables = [k for k in self.contents if k not in include]
#             else:
#                 excludables = []
#             excludables.extend([i for i in self.contents if i not in exclude])
#             new_graph = copy.deepcopy(self)
#             for node in convert.iterify(item = excludables):
#                 new_graph.delete(node = node)
#         return new_graph

#     def walk(self, 
#              start: composites.Node, 
#              stop: composites.Node, 
#              path: hybrid.Pipeline = None,
#              depth_first: bool = True) -> hybrid.Pipeline:
#         """Returns all paths in graph from 'start' to 'stop'.

#         The code here is adapted from: https://www.python.org/doc/essays/graphs/
        
#         Args:
#             start (composites.Node): node to start paths from.
#             stop (composites.Node): node to stop paths.
#             path (hybrid.Pipeline): a path from 'start' to 'stop'. Defaults to an 
#                 empty list. 

#         Returns:
#             hybrid.Pipeline: a list of possible paths (each path is a list 
#                 nodes) from 'start' to 'stop'.
            
#         """
#         if path is None:
#             path = []
#         path = path + [start]
#         if start == stop:
#             return [path]
#         if start not in self.contents:
#             return []
#         if depth_first:
#             method = self._depth_first_search
#         else:
#             method = self._breadth_first_search
#         paths = []
#         for node in self.contents[start]:
#             if node not in path:
#                 new_paths = self.walk(
#                     start = node, 
#                     stop = stop, 
#                     path = path,
#                     depth_first = depth_first)
#                 for new_path in new_paths:
#                     paths.append(new_path)
#         return paths

#     def _all_paths_bfs(self, start, stop):
#         """

#         """
#         if start == stop:
#             return [start]
#         visited = {start}
#         queue = collections.deque([(start, [])])
#         while queue:
#             current, path = queue.popleft()
#             visited.add(current)
#             for connected in self[current]:
#                 if connected == stop:
#                     return path + [current, connected]
#                 if connected in visited:
#                     continue
#                 queue.append((connected, path + [current]))
#                 visited.add(connected)   
#         return []

#     def _breadth_first_search(self, node: composites.Node) -> hybrid.Pipeline:
#         """Returns a breadth first search path through the Graph.

#         Args:
#             node (composites.Node): node to start the search from.

#         Returns:
#             hybrid.Pipeline: nodes in a path through the Graph.
            
#         """        
#         visited = set()
#         queue = [node]
#         while queue:
#             vertex = queue.pop(0)
#             if composites. not in visited:
#                 visited.add(vertex)
#                 queue.extend(set(self[vertex]) - visited)
#         return list(visited)
       
#     def _depth_first_search(self, 
#         node: composites.Node, 
#         visited: list[composites.Node]) -> hybrid.Pipeline:
#         """Returns a depth first search path through the Graph.

#         Args:
#             node (composites.Node): node to start the search from.
#             visited (list[composites.Node]): list of visited nodes.

#         Returns:
#             hybrid.Pipeline: nodes in a path through the Graph.
            
#         """  
#         if node not in visited:
#             visited.append(node)
#             for edge in self[node]:
#                 self._depth_first_search(node = edge, visited = visited)
#         return visited
  
#     def _find_all_paths(self, 
#         starts: Union[composites.Node, Sequence[composites.Node]],
#         stops: Union[composites.Node, Sequence[composites.Node]],
#         depth_first: bool = True) -> hybrid.Pipeline:
#         """[summary]

#         Args:
#             start (Union[composites.Node, Sequence[composites.Node]]): starting points for 
#                 paths through the Graph.
#             ends (Union[composites.Node, Sequence[composites.Node]]): endpoints for paths 
#                 through the Graph.

#         Returns:
#             hybrid.Pipeline: list of all paths through the Graph from all
#                 'starts' to all 'ends'.
            
#         """
#         all_paths = []
#         for start in convert.iterify(item = starts):
#             for end in convert.iterify(item = stops):
#                 paths = self.walk(
#                     start = start, 
#                     stop = end,
#                     depth_first = depth_first)
#                 if paths:
#                     if all(isinstance(path, composites.Node) for path in paths):
#                         all_paths.append(paths)
#                     else:
#                         all_paths.extend(paths)
#         return all_paths
            
#     """ Dunder Methods """

#     def __add__(self, other: Graph) -> None:
#         """Adds 'other' Graph to this Graph.

#         Adding another graph uses the 'merge' method. Read that method's 
#         docstring for further details about how the graphs are added 
#         together.
        
#         Args:
#             other (Graph): a second Graph to join with this one.
            
#         """
#         self.merge(graph = other)        
#         return

#     def __iadd__(self, other: Graph) -> None:
#         """Adds 'other' Graph to this Graph.

#         Adding another graph uses the 'merge' method. Read that method's 
#         docstring for further details about how the graphs are added 
#         together.
        
#         Args:
#             other (Graph): a second Graph to join with this one.
            
#         """
#         self.merge(graph = other)        
#         return

#     def __contains__(self, nodes: composites.Nodes) -> bool:
#         """[summary]

#         Args:
#             nodes (composites.Nodes): [description]

#         Returns:
#             bool: [description]
            
#         """
#         if isinstance(nodes, (list, tuple, set)):
#             return all(n in self.contents for n in nodes)
#         elif isinstance(nodes, composites.Node):
#             return nodes in self.contents
#         else:
#             return False   
        
#     def __getitem__(self, key: composites.Node) -> Any:
#         """Returns value for 'key' in 'contents'.

#         Args:
#             key (composites.Node): key in 'contents' for which a value is sought.

#         Returns:
#             Any: value stored in 'contents'.

#         """
#         return self.contents[key]

#     def __setitem__(self, key: composites.Node, value: Any) -> None:
#         """sets 'key' in 'contents' to 'value'.

#         Args:
#             key (composites.Node): key to set in 'contents'.
#             value (Any): value to be paired with 'key' in 'contents'.

#         """
#         self.contents[key] = value
#         return

#     def __delitem__(self, key: composites.Node) -> None:
#         """Deletes 'key' in 'contents'.

#         Args:
#             key (composites.Node): key in 'contents' to delete the key/value pair.

#         """
#         del self.contents[key]
#         return

#     def __missing__(self) -> list:
#         """Returns an empty list when a key doesn't exist.

#         Returns:
#             list: an empty list.

#         """
#         return []
    
#     def __str__(self) -> str:
#         """Returns prettier summary of the Graph.

#         Returns:
#             str: a formatted str of class information and the contained 
#                 adjacency list.
            
#         """
#         new_line = '\n'
#         tab = '    '
#         summary = [f'{new_line}piles {self.__class__.__name__}']
#         summary.append('adjacency list:')
#         for node, edges in self.contents.items():
#             summary.append(f'{tab}{node}: {str(edges)}')
#         return new_line.join(summary) 

