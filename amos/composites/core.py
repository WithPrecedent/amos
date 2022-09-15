"""
composites: extensible, flexible, lightweight complex data structures
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
    Graph (bases.Composite, ABC): base class for graphs. All subclasses 
        must have 'connect' and 'disconnect' methods for changing edges between
        nodes.
    Edge (Sequence):
    Node (object): wrapper for items that can be stored in an amos Composite
        data structure.  
    Nodes (bunches.Bunch): a collection of Node instances.
    nodify: converts any python object into a virtual Node subclass instance by 
        making it hashable, so it can be used in a Composite and pass the 
        'is_node' type check or isinstance(item, Node).
          
To Do:
    Integrate Kinds system when it is finished.
    
"""
from __future__ import annotations
import abc
from collections.abc import Collection, Sequence
import dataclasses
import inspect
from typing import Any, ClassVar, Optional, Type, TYPE_CHECKING, Union

from ..containers import bases
from ..change import convert
from ..observe import check

if TYPE_CHECKING:
    from . import forms
 
@dataclasses.dataclass # type: ignore
class Graph(bases.Composite, abc.ABC):
    """Base class for graph data structures.
    
    Args:
        contents (Collection[Any]): stored collection of nodes and/or edges.
                                      
    """  
    contents: Collection[Any]
   
    """ Required Subclass Properties """

    @abc.abstractproperty
    def adjacency(self) -> forms.Adjacency:
        """Returns the stored graph as an Adjacency."""
        pass

    @abc.abstractproperty
    def edges(self) -> forms.Edges:
        """Returns the stored graph as an Edges."""
        pass
       
    @abc.abstractproperty
    def matrix(self) -> forms.Matrix:
        """Returns the stored graph as a Matrix."""
        pass
       
    """ Required Subclass Methods """
    
    @abc.abstractclassmethod
    def from_adjacency(cls, item: forms.Adjacency) -> Graph:
        """Creates a Graph instance from an Adjacency."""
        pass
    
    @abc.abstractclassmethod
    def from_edges(cls, item: forms.Edges) -> Graph:
        """Creates a Graph instance from an Edges."""
        pass
    
    @abc.abstractclassmethod
    def from_matrix(cls, item: forms.Matrix) -> Graph:
        """Creates a Graph instance from a Matrix."""
        pass
        
    """ Dunder Methods """
    
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        """Returns whether 'instance' meets criteria to be a subclass.

        Args:
            instance (object): item to test as an instance.

        Returns:
            bool: whether 'instance' meets criteria to be a subclass.
            
        """
        return check.is_graph(item = instance)

     
@dataclasses.dataclass(frozen = True, order = True)
class Edge(Sequence):
    """Base class for an edge in a composite structure.
    
    If a subclass adds other attributes, it is important that they are not 
    declared as dataclass fields to allow indexing to work properly.
    
    Edges are not required for most of the base composite classes in amos. But
    they can be used by subclasses of those base classes for more complex data
    structures.
    
    Args:
        start (Node): starting point for the edge.
        stop (Node): stopping point for the edge.
        
    """
    start: Node
    stop: Node

    """ Dunder Methods """
        
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        """Returns whether 'instance' meets criteria to be a subclass.

        Args:
            instance (object): item to test as an instance.

        Returns:
            bool: whether 'instance' meets criteria to be a subclass.
            
        """
        return check.is_edge(item = instance)
    
    def __getitem__(self, index: int) -> Node:
        """Allows Edge subclass to be accessed by index.
        
        Args:
            index (int): the number of the field in the dataclass based on 
                order.
        
        Returns:
            Node: contents of field identified by 'index'.
                 
        """
        return getattr(self, dataclasses.fields(self)[index].name)
    
    def __len__(self) -> int:
        """Returns length based on the number of fields.
        
        Returns:
            int: number of fields.
            
        """
        return len(dataclasses.fields(self))
    
 
@dataclasses.dataclass
class Node(bases.Proxy):
    """Vertex wrapper to provide hashability to any object.
    
    Node acts a basic wrapper for any item stored in a composite structure.
    
    Args:
        contents (Optional[Any]): any stored item(s). Defaults to None.
        name (Optional[str]): designates the name of a class instance that is 
            used for internal and external referencing in a composite object.
            Defaults to None.
            
    """
    contents: Optional[Any] = None
    name: Optional[str] = None

    """ Initialization Methods """
    
    def __init_subclass__(cls, *args: Any, **kwargs: Any):
        """Forces subclasses to use the same hash methods as Node.
        
        This is necessary because dataclasses, by design, do not automatically 
        inherit the hash and equivalance dunder methods from their super 
        classes.
        
        """
        # Calls other '__init_subclass__' methods for parent and mixin classes.
        try:
            super().__init_subclass__(*args, **kwargs) # type: ignore
        except AttributeError:
            pass
        # Copies hashing related methods to a subclass.
        cls.__hash__ = Node.__hash__ # type: ignore
        cls.__eq__ = Node.__eq__ # type: ignore
        cls.__ne__ = Node.__ne__ # type: ignore

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Sets 'name' attribute if 'name' is None.
        self.name = self.name or convert.namify(item = self)
                
    """ Dunder Methods """
    
    @classmethod
    def __subclasshook__(cls, subclass: Type[Any]) -> bool:
        """Returns whether 'subclass' is a virtual or real subclass.

        Args:
            subclass (Type[Any]): item to test as a subclass.

        Returns:
            bool: whether 'subclass' is a real or virtual subclass.
            
        """
        return check.is_node(item = subclass)
               
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        """Returns whether 'instance' meets criteria to be a subclass.

        Args:
            instance (object): item to test as an instance.

        Returns:
            bool: whether 'instance' meets criteria to be a subclass.
            
        """
        return check.is_node(item = instance)
    
    def __hash__(self) -> int:
        """Makes Node hashable so that it can be used as a key in a dict.

        Rather than using the object ID, this method prevents two Nodes with
        the same name from being used in a composite object that uses a dict as
        its base storage type.
        
        Returns:
            int: hashable of 'name'.
            
        """
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Makes Node hashable so that it can be used as a key in a dict.

        Args:
            other (object): other object to test for equivalance.
            
        Returns:
            bool: whether 'name' is the same as 'other.name'.
            
        """
        try:
            return str(self.name) == str(other.name) # type: ignore
        except AttributeError:
            return str(self.name) == other

    def __ne__(self, other: object) -> bool:
        """Completes equality test dunder methods.

        Args:
            other (object): other object to test for equivalance.
           
        Returns:
            bool: whether 'name' is not the same as 'other.name'.
            
        """
        return not(self == other)

 
@dataclasses.dataclass
class Nodes(bases.Bunch):
    """Collection of Nodes.
    
    Nodes are not guaranteed to be in order. 

    Args:
        contents (Optional[Any]): any stored item(s). Defaults to None.
            
    """
    contents: Optional[Collection[Node]] = None
    
    """ Dunder Methods """ 
    
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        """Returns whether 'instance' meets criteria to be a subclass.

        Args:
            instance (object): item to test as an instance.

        Returns:
            bool: whether 'instance' meets criteria to be a subclass.
            
        """
        return check.is_nodes(item = instance)
    

def nodify(item: Union[Type[Any], object]) -> Union[Type[Node], Node]:
    """Converts a class or object into a Node for a composite data structure.
    
    Currently, the method supports any object but only python dataclass types 
    for classes. And those dataclasses should not have a '__post_init__' 
    method - it will be overwritten by 'nodify'.

    Args:
        item (Union[Type[Any], object]): class or instance to transform into a  
            Node.

    Returns:
        Union[Type[Node], Node]: a Node class or instance.
        
    """
    item.__hash__ = Node.__hash__ # type: ignore
    item.__eq__ = Node.__eq__ # type: ignore
    item.__ne__ = Node.__ne__ # type: ignore
    if inspect.isclass(item):
        item.__post_init__ = Node.__post_init__ # type: ignore
    else:
        if not hasattr(item, 'name') or not item.name:
            item.name = convert.namify(item = item)
    return item
