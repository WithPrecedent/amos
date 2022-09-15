"""
trees: tree data structures 
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2022, Corey Rayburn Yung
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

Contents:
    Tree (composites.Hybrid, composites.Composite):
    Categorizer (Tree):
    depth_first_search:
        
To Do:
    Fix 'breadth_first_search' algorithm for speed.
   
"""
from __future__ import annotations
import abc
from collections.abc import MutableSequence, Sequence
import dataclasses
from typing import Any, Callable, ClassVar, Optional, Type, TypeVar, Union

from . import core
from ..containers import sequences


""" Type Aliases """

Changer: Type[Any] = Callable[[core.Node], None]
Finder: Type[Any] = Callable[[core.Node], Optional[core.Node]]


@dataclasses.dataclass # type: ignore
class Tree(sequences.Hybrid, core.Composite, abc.ABC):
    """composites class for an tree data structures.
    
    The Tree class uses a Hybrid instead of a linked list for storing children
    nodes to allow easier access of nodes further away from the root. For
    example, a user might use 'a_tree["big_branch"]["small_branch"]["a_leaf"]' 
    to access a desired node instead of 'a_tree[2][0][3]' (although the latter
    access technique is also supported).
    
    There are several differences between a Tree and a Graph in piles:
        1) Graphs are more flexible. Trees must have 1 root, are directed, and
            each node can have only 1 parent node.
        2) Edges are only implicit in a Tree whereas they are explicit in a 
            Graph. This allows for certain methods and functions surrounding
            iteration and traversal to be faster.
        3) As the size of the data structure increases, a Tree should use less
            memory because the data about relationships between nodes is not
            centrally maintained (as with an adjacency matrix). This decreases
            access time to non-consecutive nodes, but is more efficient for 
            larger data structures.
        
    Args:
        contents (MutableSequence[Node]): list of stored Tree or other 
            Node instances. Defaults to an empty list.
        name (Optional[str]): name of Tree node which should match a parent 
            tree's key name corresponding to this Tree node. All nodes in a Tree
            must have unique names. The name is used to make all Tree nodes 
            hashable and capable of quick comparison. Defaults to None, but it
            should not be left as None when added to a Tree.
        parent (Optional[Tree]): parent Tree, if any. Defaults to None.
  
    """
    contents: MutableSequence[core.Node] = dataclasses.field(
        default_factory = list)
    name: Optional[str] = None
    parent: Optional[Tree] = None 
    
    """ Properties """
    
    @property
    def children(self) -> MutableSequence[core.Node]:
        return self.contents
    
    @children.setter
    def children(self, value: MutableSequence[core.Node]) -> None:
        self.contents = value
        return
    
    @children.deleter
    def children(self, key: core.Node) -> None:
        self.delete(key)
        return
           
    """ Dunder Methods """

    def __add__(self, other: core.Composite) -> None:
        """Adds 'other' to the stored tree using the 'append' method.

        Args:
            other (composites.Composite): another Tree, an adjacency list, an 
                edge list, an adjacency matrix, or one or more nodes.
            
        """
        self.append(item = other)     
        return 

    def __radd__(self, other: core.Composite) -> None:
        """Adds 'other' to the stored tree using the 'prepend' method.

        Args:
            other (composites.Composite): another Tree, an adjacency list, an 
                edge list, an adjacency matrix, or one or more nodes.
            
        """
        self.prepend(item = other)     
        return 

    def __missing__(self) -> dict[str, Tree]:
        """[summary]

        Returns:
            dict[str, Tree]: [description]
            
        """
        return self.__class__()
    
    def __hash__(self) -> int:
        """[summary]

        Returns:
            int: [description]
            
        """
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        """[summary]

        Args:
            other (Any): [description]

        Returns:
            bool: [description]
            
        """
        if hasattr(other, 'name'):
            return other.name == self.name
        else:
            return False
        
    def __ne__(self, other: Any) -> bool:
        """[summary]

        Args:
            other (Any): [description]

        Returns:
            bool: [description]
            
        """
        return not self.__eq__(other = other)


@dataclasses.dataclass # type: ignore
class Categorizer(Tree):
    """composites class for an tree data structures.
        
    Args:
        contents (MutableSequence[composites.Node]): list of stored Node 
            instances (including other Trees). Defaults to an empty list.
        name (Optional[str]): name of Tree node which should match a parent 
            tree's key name corresponding to this Tree node. All nodes in a Tree
            must have unique names. The name is used to make all Tree nodes 
            hashable and capable of quick comparison. Defaults to None, but it
            should not be left as None when added to a Tree.
        parent (Optional[Tree]): parent Tree, if any. Defaults to None.
        
    """
    contents: MutableSequence[core.Node] = dataclasses.field(
        default_factory = list)
    name: Optional[str] = None
    parent: Optional[Tree] = None 
    
    """ Properties """
        
    @property
    def branches(self) -> list[Tree]:
        """Returns all stored Tree nodes in a list."""
        return self.nodes - self.leaves
    
    @property
    def children(self) -> dict[str, core.Node]:
        """[summary]

        Returns:
            dict[str, composites.Node]: [description]
        """
        return self.contents
    
    @property
    def is_leaf(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return not self.children
    
    @property
    def is_root(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return self.parent is None
    
    @property
    def leaves(self) -> list[core.Node]:
        """Returns all stored leaf nodes in a list."""
        matches = []
        for node in self.nodes:
            if not hasattr(node, 'is_leaf') or node.is_leaf:
                matches.append(node)
        return matches
     
    @property
    def nodes(self) -> list[core.Node]:
        """Returns all stored nodes in a list."""
        return depth_first_search(tree = self.contents)

    @property
    def root(self) -> Tree:
        """
        """
        composites = [n.is_root for n in self.nodes]
        if len(composites) > 1:
            raise ValueError('The tree is broken - it has more than 1 root')
        elif len(composites) == 0:
            raise ValueError('The tree is broken - it has no root')
        else:
            return composites[0]
    
    """ Public Methods """
    
    def add(
        self, 
        item: Union[core.Node, Sequence[core.Node]],
        parent: Optional[str] = None) -> None:
        """Adds node(s) in item to 'contents'.
        
        In adding the node(s) to the stored tree, the 'parent' attribute for the
        node(s) is set to this Tree instance.

        Args:
            item (Union[composites.Node, Sequence[composites.Node]]): node(s) to 
                add to the 'contents' attribute.

        Raises:
            ValueError: if 'item' already is in the stored tree or if 'parent'
                is not in the tree.
                            
        """
        if parent is None:
            parent_node = self
        else:
            parent_node = self.get(item = parent)
        if parent_node is None:
            raise ValueError(
                f'Cannot add {item.name} because parent node {parent} is not '
                f'in the tree')
        if isinstance(item, Sequence) and not isinstance(item, str):
            for node in item:
                self.add(item = node)
        elif item in self.nodes:
            raise ValueError(
                f'Cannot add {item.name} because it is already in the tree')
        else:
            item.parent = parent_node
            parent_node.contents.append(item)
        return
    
    def find(self, finder: Finder, **kwargs: Any) -> Optional[core.Node]:
        """Finds first matching node in Tree using 'finder'.

        Args:
            finder (Callable[[composites.Node], Optional[composites.Node]]): 
                function or other callable that returns a node if it meets 
                certain criteria or otherwise returns None.
            kwargs: keyword arguments to pass to 'finder' when examing each
                node.

        Returns:
            Optional[composites.Node]: matching Node or None if no matching node 
                is found.
            
        """                  
        for node in self.nodes:
            comparison = finder(self, **kwargs)
            if comparison:
                return node
        return None
            
    def find_add(
        self, 
        finder: Finder, 
        item: core.Node, 
        **kwargs: Any) -> None:
        """Finds first matching node in Tree using 'finder'.

        Args:
            finder (Callable[[composites.Node], Optional[composites.Node]]): 
                function or other callable that returns a node if it meets 
                certain criteria or otherwise returns None.
            item (composites.Node): node to add to the 'contents' attribute of 
                the first node that meets criteria in 'finder'.
            kwargs: keyword arguments to pass to 'finder' when examing each
                node.

        Raises:
            ValueError: if no matching node is found by 'finder'.

        Returns:
            Optional[composites.Node]: matching Node or None if no matching node 
                is found.
            
        """  
        node = self.find(finder = finder, **kwargs)
        if node:
            node.add(item = item)
        else:
            raise ValueError(
                'item could not be added because no matching node was found by '
                'finder')
        return
    
    def find_all(self, finder: Finder, **kwargs: Any) -> list[core.Node]:
        """Finds all matching nodes in Tree using 'finder'.

        Args:
            finder (Callable[[composites.Node], Optional[composites.Node]]): 
                function or other callable that returns a node if it meets 
                certain criteria or otherwise returns None.
            kwargs: keyword arguments to pass to 'finder' when examing each
                node.

        Returns:
            list[composites.Node]: matching nodes or an empty list if no 
                matching node is found.
            
        """              
        found = []     
        for node in self.nodes:
            comparison = finder(self, **kwargs)
            if comparison:
                found.append(node)
        return found
            
    def find_change(
        self, 
        finder: Finder, 
        changer: Changer, 
        **kwargs: Any) -> None:
        """Finds matching nodes in Tree using 'finder' and applies 'changer'.

        Args:
            finder (Callable[[composites.Node], Optional[composites.Node]]): 
                function or other callable that returns a node if it meets 
                certain criteria or otherwise returns None.
            changer (Callable[[composites.Node], None]): function or other 
                callable that modifies the found node.
            kwargs: keyword arguments to pass to 'finder' when examing each
                node.

        Raises:
            ValueError: if no matching node is found by 'finder'.
            
        """  
        nodes = self.find_all(finder = finder, **kwargs)
        if nodes:
            for node in nodes:
                changer(node)
        else:
            raise ValueError(
                'changer could not be applied because no matching node was '
                'found by finder')
        return
    
    def get(self, item: str) -> Optional[core.Node]:
        """Finds first matching node in Tree match 'item'.

        Args:
            item (str): 

        Returns:
            Optional[composites.Node]: matching Node or None if no matching node 
                is found.
            
        """                  
        for node in self.nodes:
            if node.name == item:
                return node
        return self.__missing__()
                                    
    def walk(self, depth_first: bool = True) -> core.Pipeline:
        """Returns all paths in tree from 'start' to 'stop'.
        
        Args:
            depth_first (bool): whether to search through the stored tree depth-
                first (True) or breadth_first (False). Defaults to True.
                
        """
        if depth_first:
            return depth_first_search(tree = self.contents)
        else:
            raise NotImplementedError(
                'breadth first search is not yet implemented')
            # return breadth_first_search(tree = self.contents)

    """ Dunder Methods """

    def __add__(self, other: core.Composite) -> None:
        """Adds 'other' to the stored tree using the 'append' method.

        Args:
            other (composites.Composite): another Composite or supported
                raw data structure.
            
        """
        self.append(item = other)     
        return 

    def __radd__(self, other: core.Composite) -> None:
        """Adds 'other' to the stored tree using the 'prepend' method.

        Args:
            other (composites.Composite): another Composite or supported
                raw data structure.
            
        """
        self.prepend(item = other)     
        return 

    def __missing__(self) -> dict[str, Tree]:
        """[summary]

        Returns:
            dict[str, Tree]: [description]
            
        """
        return {}
    
    def __hash__(self) -> int:
        """[summary]

        Returns:
            int: [description]
            
        """
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        """[summary]

        Args:
            other (Any): [description]

        Returns:
            bool: [description]
            
        """
        if hasattr(other, 'name'):
            return other.name == self.name
        else:
            return False
        
    def __ne__(self, other: Any) -> bool:
        """[summary]

        Args:
            other (Any): [description]

        Returns:
            bool: [description]
            
        """
        return not self.__eq__(other = other)


# def breadth_first_search(
#     tree: Tree, 
#     visited: Optional[list[Tree]] = None) -> composites.Pipeline:
#     """Returns a breadth first search path through 'tree'.

#     Args:
#         tree (Tree): tree to search.
#         visited (Optional[list[Tree]]): list of visited nodes. Defaults to None.

#     Returns:
#         composites.Pipeline: nodes in a path through 'tree'.
        
#     """         
#     visited = visited or []
#     if hasattr(tree, 'is_root') and tree.is_root:
#         visited.append(tree)
#     if hasattr(tree, 'children') and tree.children:
#         visited.extend(tree.children)
#         for child in tree.children:
#             visited.extend(breadth_first_search(tree = child, visited = visited))
#     return visited
                
                     
def depth_first_search(
    tree: Tree, 
    visited: Optional[list[Tree]] = None) -> core.Pipeline:
    """Returns a depth first search path through 'tree'.

    Args:
        tree (Tree): tree to search.
        visited (Optional[list[Tree]]): list of visited nodes. Defaults to None.

    Returns:
        composites.Pipeline: nodes in a path through 'tree'.
        
    """  
    visited = visited or []
    visited.append(tree)
    if hasattr(tree, 'children') and tree.children:
        for child in tree.children:
            visited.extend(depth_first_search(tree = child, visited = visited))
    return visited

