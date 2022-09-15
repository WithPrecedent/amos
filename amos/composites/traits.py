"""
traits: characteristics of amos graphs
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
    Directed (Adjacency): a directed graph with unweighted edges.
        
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

from . import core
from . import forms
from . import hybrids
from ..containers import mappings
from ..observe import report
from ..change import convert
from ..observe import check
from ..change import convert

if TYPE_CHECKING:
    from . import trees
    
    
@dataclasses.dataclass
class Directed(abc.ABC):
    """Base class for directed graph data structures."""  
    
    """ Required Subclass Properties """
        
    @abc.abstractproperty
    def endpoint(self) -> Optional[Union[Hashable, Collection[Hashable]]]:
        """Returns the endpoint(s) of the stored composite object."""
        pass
 
    @abc.abstractproperty
    def root(self) -> Optional[Union[Hashable, Collection[Hashable]]]:
        """Returns the root(s) of the stored composite object."""
        pass
        
    @abc.abstractproperty
    def pipeline(self) -> hybrids.Pipeline:
        """Returns the stored composite object as a hybrids.Pipeline."""
        pass
        
    @abc.abstractproperty
    def pipelines(self) -> hybrids.Pipelines:
        """Returns the stored composite object as a hybrids.Pipelines."""
        pass
            
    @abc.abstractproperty
    def tree(self) -> tree.Tree:
        """Returns the stored composite object as a tree.Tree."""
        pass
                 
    """ Required Subclass Class Methods """
    
    @abc.abstractclassmethod
    def from_pipeline(cls, item: hybrids.Pipeline) -> Directed:
        """Creates a Composite instance from a hybrids.Pipeline."""
        pass
    
    @abc.abstractclassmethod
    def from_pipelines(cls, item: hybrids.Pipelines) -> Directed:
        """Creates a Composite instance from a hybrids.Pipelines."""
        pass

    @abc.abstractclassmethod
    def from_tree(cls, item: tree.Tree) -> Directed:
        """Creates a Composite instance from a tree.Tree."""
        pass
                 
    """ Required Subclass Methods """
    
    @abc.abstractmethod
    def add(self, item: Hashable, *args: Any, **kwargs: Any) -> None:
        """Adds 'Hashable' to the stored composite object.
        
        Args:
            Hashable (Hashable): a Hashable to add to the stored composite object.
                
        """
        pass
    
    @abc.abstractmethod
    def append(
        self, 
        item: Union[Hashable, Collection[Hashable], core.Graph], 
        *args: Any, 
        **kwargs: Any) -> None:
        """Appends 'item' to the endpoint(s) of the stored composite object.

        Args:
            item (Union[Hashable, core.Graph]): a single Hashable or other Composite
                object to add to the stored composite object.
                
        """
        pass
        
    @abc.abstractmethod
    def delete(self, item: Any, *args: Any, **kwargs: Any) -> None:
        """Deletes Hashable from the stored composite object.
        
        Args:
            item (Any): Hashable or key to the a Hashable to delete.
        
        Raises:
            KeyError: if 'item' is not in 'contents'.
            
        """
        pass
  
    @abc.abstractmethod
    def merge(self, item: core.Graph, *args: Any, **kwargs: Any) -> None:
        """Combines 'item' with the stored composite object.

        Args:
            item (core.Graph): another Composite object to add to the stored 
                composite object.
                
        """
        pass
    
    @abc.abstractmethod
    def prepend(
        self, 
        item: Union[Hashable, Collection[Hashable], core.Graph], 
        *args: Any, 
        **kwargs: Any) -> None:
        """Prepends 'item' to the root(s) of the stored composite object.

        Args:
            item (Union[Hashable, core.Graph]): a single Hashable or other Composite
                object to add to the stored composite object.
                
        """
        pass
    
    @abc.abstractmethod
    def subset(
        self, 
        include: Union[Any, Sequence[Any]] = None,
        exclude: Union[Any, Sequence[Any]] = None, 
        *args: Any, 
        **kwargs: Any) -> core.Graph:
        """Returns a new Composite with a subset of 'contents'.
        
        Args:
            include (Union[Any, Sequence[Any]]): Collection[Hashable] which should be included
                in the new Composite.
            exclude (Union[Any, Sequence[Any]]): Collection[Hashable] which should not be 
                in the new Composite.

        Returns:
           Composite: with only Collection[Hashable] indicated by 'include' and 'exclude'.
           
        """
        pass
    
    @abc.abstractmethod
    def walk(
        self, 
        start: Optional[Hashable] = None,
        stop: Optional[Hashable] = None, 
        path: Optional[hybrids.Pipeline] = None,
        return_pipelines: bool = True, 
        *args: Any, 
        **kwargs: Any) -> Union[hybrids.Pipeline, hybrids.Pipelines]:
        """Returns path in the stored composite object from 'start' to 'stop'.
        
        Args:
            start (Optional[Hashable]): Hashable to start paths from. Defaults to None.
                If it is None, 'start' should be assigned to one of the roots
                of the Composite.
            stop (Optional[Hashable]): Hashable to stop paths. Defaults to None. If it 
                is None, 'start' should be assigned to one of the roots of the 
                Composite.
            path (Optional[hybrids.Pipeline]): a path from 'start' to 'stop'. 
                Defaults to None. This parameter is used by recursive methods 
                for determining a path.
            return_pipelines (bool): whether to return a Pipelines instance 
                (True) or a hybrids.Pipeline instance (False). Defaults to True.

        Returns:
            Union[hybrids.Pipeline, hybrids.Pipelines]: path(s) through the 
                Composite object. If multiple paths are possible and 
                'return_pipelines' is False, this method should return a 
                Pipeline that includes all such paths appended to each other. If 
                multiple paths are possible and 'return_pipelines' is True, a 
                Pipelines instance with all of the paths should be returned. 
                Defaults to True.
                            
        """
        pass
    
    """ Dunder Methods """

    def __add__(self, other: core.Graph) -> None:
        """Adds 'other' to the stored composite object using 'append'.

        Args:
            other (Union[base.Composite]): another Composite subclass instance 
                to add to the current one.
            
        """
        self.append(item = other)     
        return 

    def __radd__(self, other: core.Graph) -> None:
        """Adds 'other' to the stored composite object using 'prepend'.

        Args:
            other (Union[base.Composite]): another Composite subclass instance 
                to add to the current one.
            
        """
        self.prepend(item = other)     
        return 

    # def __str__(self) -> str:
    #     """Returns prettier str representation of an instance.

    #     Returns:
    #         str: a formatted str of an instance.
            
    #     """
    #     return represent.beautify(item = self, package = 'amos')  
    
    