"""
composites: base classes for extensible, flexible, lightweight data structures
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
    Composite (factories.StealthFactory, bunches.Bunch): base class for amos
        composite data structures. In addition to the 'add' and 'subset' methods
        required by inheriting from Bunch, Composite requires 'append', 
        'delete', 'merge', 'prepend', and 'walk' methods.
    Node (object): wrapper for items that can be stored in an amos Composite
        data structure.  
    Nodes (bunches.Bunch): a collection of Node instances.
    nodify: converts any python object into a Node by making it hashable, so it 
        can be used in a Composite and pass the 'is_node' type check.
          
To Do:
    Integrate ashford Kinds system when it is finished.
    Add in 'beautify' str representations from amos once those are finished.
    Add universal 'merge' method in Composite to replace the ad hoc methods of
        subclasses. This requires completion of "To Do:" tasks in the 'convert'
        module as well.
    
"""
from __future__ import annotations
import abc
from collections.abc import Collection, Hashable, Sequence
import dataclasses
import inspect
from typing import Any, ClassVar, Optional, Type, TYPE_CHECKING, Union

from . import bases
from ..change import convert
from ..observe import check


if TYPE_CHECKING:
    from . import hybrid
    from . import sequences

          
@dataclasses.dataclass
class Composite(abc.ABC):
    """Base class for composite data structures.
    
    Args:
        contents (Collection[Any]): stored collection of nodes and/or edges.
                                     
    """  
    contents: Collection[Any]
    
    """ Required Subclass Properties """
        
    @abc.abstractproperty
    def endpoint(self) -> Optional[Union[Node, Nodes]]:
        """Returns the endpoint(s) of the stored composite object."""
        pass
 
    @abc.abstractproperty
    def root(self) -> Optional[Union[Node, Nodes]]:
        """Returns the root(s) of the stored composite object."""
        pass

    # @abc.abstractproperty
    # def adjacency(self) -> graph.Adjacency:
    #     """Returns the stored composite object as an graph.Adjacency."""
    #     pass

    # @abc.abstractproperty
    # def edges(self) -> graph.Edges:
    #     """Returns the stored composite object as an graph.Edges."""
    #     pass
       
    # @abc.abstractproperty
    # def matrix(self) -> graph.Matrix:
    #     """Returns the stored composite object as a graph.Matrix."""
    #     pass
       
    # @abc.abstractproperty
    # def nodes(self) -> Nodes:
    #     """Returns the stored composite object as a Nodes."""
    #     pass
        
    # @abc.abstractproperty
    # def pipeline(self) -> hybrid.Pipeline:
    #     """Returns the stored composite object as a hybrid.Pipeline."""
    #     pass
        
    # @abc.abstractproperty
    # def pipelines(self) -> hybrid.Pipelines:
    #     """Returns the stored composite object as a hybrid.Pipelines."""
    #     pass
            
    # @abc.abstractproperty
    # def tree(self) -> tree.Tree:
    #     """Returns the stored composite object as a tree.Tree."""
    #     pass
                 
    # """ Required Subclass Class Methods """
    
    # @abc.abstractclassmethod
    # def from_adjacency(cls, item: graph.Adjacency) -> Composite:
    #     """Creates a Composite instance from an graph.Adjacency."""
    #     pass
    
    # @abc.abstractclassmethod
    # def from_edges(cls, item: graph.Edges) -> Composite:
    #     """Creates a Composite instance from an graph.Edges."""
    #     pass
    
    # @abc.abstractclassmethod
    # def from_matrix(cls, item: graph.Matrix) -> Composite:
    #     """Creates a Composite instance from a graph.Matrix."""
    #     pass
    
    # @abc.abstractclassmethod
    # def from_pipeline(cls, item: hybrid.Pipeline) -> Composite:
    #     """Creates a Composite instance from a hybrid.Pipeline."""
    #     pass
    
    # @abc.abstractclassmethod
    # def from_pipelines(cls, item: hybrid.Pipelines) -> Composite:
    #     """Creates a Composite instance from a hybrid.Pipelines."""
    #     pass

    # @abc.abstractclassmethod
    # def from_tree(cls, item: tree.Tree) -> Composite:
    #     """Creates a Composite instance from a tree.Tree."""
    #     pass
                 
    """ Required Subclass Methods """
    
    @abc.abstractmethod
    def add(self, item: Node, *args: Any, **kwargs: Any) -> None:
        """Adds 'node' to the stored composite object.
        
        Args:
            node (Node): a node to add to the stored composite object.
                
        """
        pass
    
    @abc.abstractmethod
    def append(
        self, 
        item: Union[Node, Nodes, Composite], 
        *args: Any, 
        **kwargs: Any) -> None:
        """Appends 'item' to the endpoint(s) of the stored composite object.

        Args:
            item (Union[Node, Composite]): a single Node or other Composite
                object to add to the stored composite object.
                
        """
        pass
        
    @abc.abstractmethod
    def delete(self, item: Any, *args: Any, **kwargs: Any) -> None:
        """Deletes node from the stored composite object.
        
        Args:
            item (Any): node or key to the a node to delete.
        
        Raises:
            KeyError: if 'item' is not in 'contents'.
            
        """
        pass
  
    @abc.abstractmethod
    def merge(self, item: Composite, *args: Any, **kwargs: Any) -> None:
        """Combines 'item' with the stored composite object.

        Args:
            item (Composite): another Composite object to add to the stored 
                composite object.
                
        """
        pass
    
    @abc.abstractmethod
    def prepend(
        self, 
        item: Union[Node, Nodes, Composite], 
        *args: Any, 
        **kwargs: Any) -> None:
        """Prepends 'item' to the root(s) of the stored composite object.

        Args:
            item (Union[Node, Composite]): a single Node or other Composite
                object to add to the stored composite object.
                
        """
        pass
    
    @abc.abstractmethod
    def subset(
        self, 
        include: Union[Any, Sequence[Any]] = None,
        exclude: Union[Any, Sequence[Any]] = None, 
        *args: Any, 
        **kwargs: Any) -> Composite:
        """Returns a new Composite with a subset of 'contents'.
        
        Args:
            include (Union[Any, Sequence[Any]]): nodes which should be included
                in the new Composite.
            exclude (Union[Any, Sequence[Any]]): nodes which should not be 
                in the new Composite.

        Returns:
           Composite: with only nodes indicated by 'include' and 'exclude'.
           
        """
        pass
    
    @abc.abstractmethod
    def walk(
        self, 
        start: Optional[Node] = None,
        stop: Optional[Node] = None, 
        path: Optional[hybrid.Pipeline] = None,
        return_pipelines: bool = True, 
        *args: Any, 
        **kwargs: Any) -> Union[hybrid.Pipeline, hybrid.Pipelines]:
        """Returns path in the stored composite object from 'start' to 'stop'.
        
        Args:
            start (Optional[Node]): node to start paths from. Defaults to None.
                If it is None, 'start' should be assigned to one of the roots
                of the Composite.
            stop (Optional[Node]): node to stop paths. Defaults to None. If it 
                is None, 'start' should be assigned to one of the roots of the 
                Composite.
            path (Optional[hybrid.Pipeline]): a path from 'start' to 'stop'. 
                Defaults to None. This parameter is used by recursive methods 
                for determining a path.
            return_pipelines (bool): whether to return a Pipelines instance 
                (True) or a hybrid.Pipeline instance (False). Defaults to True.

        Returns:
            Union[hybrid.Pipeline, hybrid.Pipelines]: path(s) through the 
                Composite object. If multiple paths are possible and 
                'return_pipelines' is False, this method should return a 
                Pipeline that includes all such paths appended to each other. If 
                multiple paths are possible and 'return_pipelines' is True, a 
                Pipelines instance with all of the paths should be returned. 
                Defaults to True.
                            
        """
        pass
    
    """ Dunder Methods """

    def __add__(self, other: Composite) -> None:
        """Adds 'other' to the stored composite object using 'append'.

        Args:
            other (Union[base.Composite]): another Composite subclass instance 
                to add to the current one.
            
        """
        self.append(item = other)     
        return 

    def __radd__(self, other: Composite) -> None:
        """Adds 'other' to the stored composite object using 'prepend'.

        Args:
            other (Union[base.Composite]): another Composite subclass instance 
                to add to the current one.
            
        """
        self.prepend(item = other)     
        return 

    # def __str__(self) -> str:
    #     """Returns prettier str representation of the stored graph.

    #     Returns:
    #         str: a formatted str of class information and the contained graph.
            
    #     """
    #     return recap.beautify(item = self, package = 'piles')  


@dataclasses.dataclass
class Node(object):
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
        """Returns whether 'instance' meets criteria to be a node.

        Args:
            instance (object): item to test as an instance.

        Returns:
            bool: whether 'instance' is a node.
            
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

    def __contains__(self, item: Any) -> bool:
        """Returns whether 'item' is in or equal to 'contents'.

        Args:
            item (Any): item to check versus 'contents'
            
        Returns:
            bool: if 'item' is in or equal to 'contents' (True). Otherwise, it
                returns False.

        """
        try:
            return item in self.contents
        except TypeError:
            try:
                return item is self.contents
            except TypeError:
                return item == self.contents
                
    def __getattr__(self, attribute: str) -> Any:
        """Looks for 'attribute' in 'contents'.

        Args:
            attribute (str): name of attribute to return.

        Raises:
            AttributeError: if 'attribute' is not found in 'contents'.

        Returns:
            Any: matching attribute.

        """
        try:
            return object.__getattribute__(
                object.__getattribute__(self, 'contents'), attribute)
        except AttributeError:
            raise AttributeError(
                f'{attribute} was not found in '
                f'{object.__getattribute__(self, "__name__")}') 

    def __setattr__(self, attribute: str, value: Any) -> None:
        """Sets 'attribute' to 'value'.
        
        If 'attribute' exists in this class instance, its new value is set to
        'value.' Otherwise, 'attribute' and 'value' are set in what is stored
        in 'contents'

        Args:
            attribute (str): name of attribute to set.
            value (Any): value to store in 'attribute'.

        """
        if hasattr(self, attribute) or self.contents is None:
            object.__setattr__(self, attribute, value)
        else:
            object.__setattr__(self.contents, attribute, value)
            
    def __delattr__(self, attribute: str) -> None:
        """Deletes 'attribute'.
        
        If 'attribute' exists in this class instance, it is deleted. Otherwise, 
        this method attempts to delete 'attribute' from what is stored in 
        'contents'

        Args:
            attribute (str): name of attribute to set.

        Raises:
            AttributeError: if 'attribute' is neither found in a class instance
                nor in 'contents'.
            
        """
        try:
            object.__delattr__(self, attribute)
        except AttributeError:
            try:
                object.__delattr__(self.contents, attribute)
            except AttributeError:
                raise AttributeError(
                    f'{attribute} was not found in '
                    f'{object.__getattribute__(self, "__name__")}') 
 
 
@dataclasses.dataclass
class Edge(object):
    """Base class for an edge in a composite structure."""

    start: Node
    stop: Node
    
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        """Checks if 'instance' is a virtual or actual subclass instance."""
        return check.is_edge(item = instance)

 
@dataclasses.dataclass
class Nodes(bases.Bunch):
    """Collection of nodes.
    
    Nodes are not guaranteed to be in order. 

    Args:
        contents (Optional[Any]): any stored item(s). Defaults to None.
            
    """
    contents: Optional[Collection[Node]] = None
    
    """ Dunder Methods """ 
    
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_nodes(item = instance)
    

def nodify(item: Union[Type[Any], object]) -> Union[Type[Node], Node]:
    """Converts a class or object into a node for a composite data structure.
    
    Currently, the method supports any object but only python dataclass types 
    for classes. And those dataclasses should not have a '__post_init__' 
    method - it will be overwritten by 'nodify'.

    Args:
        item (Union[Type[Any], object]): class or instance to transform into a  
            node.

    Returns:
        Union[Type[Node], Node]: a node class or instance.
        
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

 
@dataclasses.dataclass # type: ignore
class Pipeline(sequences.Hybrid, Composite):
    """Base class for pipeline data structures.
    
    Args:
        contents (MutableSequence[Node]): list of stored Node instances. 
            Defaults to an empty list.
          
    """
    contents: MutableSequence[Node] = dataclasses.field(
        default_factory = list)

    """ Properties """

    def endpoint(self) -> Node:
        """Returns the endpoint of the stored composite object."""
        return self.contents[-1]

    def root(self) -> Node:
        """Returns the root of the stored composite object."""
        return self.contents[0]
    
    """ Public Methods """
        
    def delete(self, item: Any) -> None:
        """Deletes node from the stored composite object.
        
        Args:
            item (Any): node or key to the a node to delete.
        
        Raises:
            KeyError: if 'item' is not in 'contents'.
            
        """
        self.__delitem__(item)
  
    def merge(item: Composite, *args: Any, **kwargs: Any) -> None:
        """Combines 'item' with the stored composite object.

        Args:
            item (Composite): another Composite object to add to the stored 
                composite object.
                
        """
        pass
   
    def walk(
        self, 
        start: Optional[Node] = None,
        stop: Optional[Node] = None, 
        path: Optional[Pipeline] = None,
        return_pipelines: bool = False, 
        *args: Any, 
        **kwargs: Any) -> Pipeline:
        """Returns path in the stored composite object from 'start' to 'stop'.
        
        Args:
            start (Optional[Node]): node to start paths from. Defaults to None.
                If it is None, 'start' should be assigned to one of the roots
                of the Composite.
            stop (Optional[Node]): node to stop paths. Defaults to None. If it 
                is None, 'start' should be assigned to one of the roots of the 
                Composite.
            path (Optional[hybrid.Pipeline]): a path from 'start' to 'stop'. 
                Defaults to None. This parameter is used by recursive methods 
                for determining a path.
            return_pipelines (bool): whether to return a Pipelines instance 
                (True) or a hybrid.Pipeline instance (False). Defaults to True.

        Returns:
            Union[hybrid.Pipeline, hybrid.Pipelines]: path(s) through the 
                Composite object. If multiple paths are possible and 
                'return_pipelines' is False, this method should return a 
                Pipeline that includes all such paths appended to each other. If 
                multiple paths are possible and 'return_pipelines' is True, a 
                Pipelines instance with all of the paths should be returned. 
                Defaults to True.
                            
        """
        return self.contents
    
    """ Dunder Methods """
        
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_pipeline(item = instance)
     
 
@dataclasses.dataclass # type: ignore
class Pipelines(sequences.Hybrid, Composite):
    """Base class a collection of Pipeline instances.
        
    Args:
        contents (MutableSequence[Node]): list of stored Pipeline instances. 
            Defaults to an empty list.

    """
    contents: MutableSequence[Pipeline] = dataclasses.field(
        default_factory = list)

    """ Properties """

    def endpoint(self) -> Pipeline:
        """Returns the endpoint of the stored composite object."""
        return self.contents[list(self.contents.keys())[-1]]

    def root(self) -> Pipeline:
        """Returns the root of the stored composite object."""
        self.contents[list(self.contents.keys())[0]]
    
    """ Public Methods """
  
    def merge(item: Composite, *args: Any, **kwargs: Any) -> None:
        """Combines 'item' with the stored composite object.

        Args:
            item (Composite): another Composite object to add to the stored 
                composite object.
                
        """
        pass

    def walk(
        self, 
        start: Optional[Node] = None,
        stop: Optional[Node] = None, 
        path: Optional[Pipeline] = None,
        return_pipelines: bool = True, 
        *args: Any, 
        **kwargs: Any) -> Union[Pipeline, Pipelines]:
        """Returns path in the stored composite object from 'start' to 'stop'.
        
        Args:
            start (Optional[Node]): node to start paths from. Defaults to None.
                If it is None, 'start' should be assigned to one of the roots
                of the Composite.
            stop (Optional[Node]): node to stop paths. Defaults to None. If it 
                is None, 'start' should be assigned to one of the roots of the 
                Composite.
            path (Optional[hybrid.Pipeline]): a path from 'start' to 'stop'. 
                Defaults to None. This parameter is used by recursive methods 
                for determining a path.
            return_pipelines (bool): whether to return a Pipelines instance 
                (True) or a hybrid.Pipeline instance (False). Defaults to True.

        Returns:
            Union[hybrid.Pipeline, hybrid.Pipelines]: path(s) through the 
                Composite object. If multiple paths are possible and 
                'return_pipelines' is False, this method should return a 
                Pipeline that includes all such paths appended to each other. If 
                multiple paths are possible and 'return_pipelines' is True, a 
                Pipelines instance with all of the paths should be returned. 
                Defaults to True.
                            
        """
        return self.items()
        
    """ Dunder Methods """
        
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_pipelines(item = instance)
         