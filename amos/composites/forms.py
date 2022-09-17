"""
forms: graphs with different internal storage formats
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
    Adjacency (mappings.Dictionary, core.Graph): a graph stored as an adjacency 
        list.
    Edges (sequences.Listing, core.Graph): a graph stored as an edge list.
    Matrix (sequences.Listing, core.Graph): a graph stored as an adjacency 
        matrix.
          
To Do:
    Integrate Kinds system when it is finished.
    
"""
from __future__ import annotations
import abc
import collections
from collections.abc import (
    Collection, Hashable, MutableMapping, MutableSequence, Sequence, Set)
import dataclasses
from typing import Any, ClassVar, Optional, Type, TYPE_CHECKING, Union

from . import core
from ..containers import mappings
from ..containers import sequences
from ..change import convert
from ..observe import check


@dataclasses.dataclass
class Adjacency(mappings.Dictionary, core.Graph):
    """Base class for adjacency-list graphs.
    
    Args:
        contents (MutableMapping[core.Node, Set[core.Node]]): keys are nodes and 
            values are sets of nodes (or hashable representations of nodes). 
            Defaults to a defaultdict that has a set for its value type.
                                      
    """  
    contents: MutableMapping[core.Node, Set[core.Node]] = dataclasses.field(
        default_factory = lambda: collections.defaultdict(set))
   
    """ Properties """

    @abc.abstractproperty
    def adjacency(self) -> Adjacency:
        """Returns the stored graph as an Adjacency."""
        return self.contents

    @abc.abstractproperty
    def edges(self) -> Edges:
        """Returns the stored graph as an Edges."""
        return convert.adjacency_to_edges(item = self.contents)
       
    @abc.abstractproperty
    def matrix(self) -> Matrix:
        """Returns the stored graph as a Matrix."""
        return convert.adjacency_to_matrix(item = self.contents)
       
    """ Class Methods """
    
    @abc.abstractclassmethod
    def from_adjacency(cls, item: Adjacency) -> Adjacency:
        """Creates an Adjacency instance from an Adjacency."""
        return cls(contents = item)
    
    @abc.abstractclassmethod
    def from_edges(cls, item: Edges) -> Adjacency:
        """Creates an Adjacency instance from an Edges."""
        return cls(contents = convert.edges_to_adjacency(item = item))
    
    @abc.abstractclassmethod
    def from_matrix(cls, item: Matrix) -> Adjacency:
        """Creates an Adjacency instance from a Matrix."""
        return cls(contents = convert.matrix_to_adjacency(item = item))
        
    """ Dunder Methods """
    
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        """Returns whether 'instance' meets criteria to be a subclass.

        Args:
            instance (object): item to test as an instance.

        Returns:
            bool: whether 'instance' meets criteria to be a subclass.
            
        """
        return check.is_adjacency(item = instance)


@dataclasses.dataclass
class Edges(sequences.Listing, core.Graph):
    """Base class for edge-list graphs.
    
    Args:
        contents (tuple[tuple[core.Node, core.Node], ...]): tuple of tuple of 
            edges. Defaults to an empty tuple.
                                      
    """   
    contents: MutableSequence[core.Edge] = dataclasses.field(
        default_factory = list)
   
    """ Properties """

    @abc.abstractproperty
    def adjacency(self) -> Adjacency:
        """Returns the stored graph as an Adjacency."""
        return convert.edges_to_adjacency(item = self.contents)

    @abc.abstractproperty
    def edges(self) -> Edges:
        """Returns the stored graph as an Edges."""
        return self.contents
       
    @abc.abstractproperty
    def matrix(self) -> Matrix:
        """Returns the stored graph as a Matrix."""
        return convert.edges_to_matrix(item = self.contents)
       
    """ Class Methods """
    
    @abc.abstractclassmethod
    def from_adjacency(cls, item: Adjacency) -> Edges:
        """Creates an Edges instance from an Adjacency."""
        return cls(contents = convert.adjacency_to_edges(item = item))
    
    @abc.abstractclassmethod
    def from_edges(cls, item: Edges) -> Edges:
        """Creates an Edges instance from an Edges."""
        
        return cls(contents = item)
    
    @abc.abstractclassmethod
    def from_matrix(cls, item: Matrix) -> Edges:
        """Creates an Edges instance from a Matrix."""
        return cls(contents = convert.adjacency_to_matrix(item = item))
        
    """ Dunder Methods """
           
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        """Returns whether 'instance' meets criteria to be a subclass.

        Args:
            instance (object): item to test as an instance.

        Returns:
            bool: whether 'instance' meets criteria to be a subclass.
            
        """
        return check.is_edges(item = instance)
    
    
@dataclasses.dataclass
class Matrix(sequences.Listing, core.Graph):
    """Base class for adjacency-matrix graphs.
    
    Args:
        contents (Sequence[Sequence[int]]): a list of list of integers 
            indicating edges between nodes in the matrix. Defaults to an empty
            list.
        labels (Sequence[Hashable]): names of nodes in the matrix. 
            Defaults to an empty list.
                                      
    """  
    contents: MutableSequence[MutableSequence[int]] = dataclasses.field(
        default_factory = list)
    labels: MutableSequence[Hashable] = dataclasses.field(
        default_factory = list)
   
    """ Properties """

    @abc.abstractproperty
    def adjacency(self) -> Adjacency:
        """Returns the stored graph as an Adjacency."""
        return convert.matrix_to_adjacency(item = self.contents)

    @abc.abstractproperty
    def edges(self) -> Edges:
        """Returns the stored graph as an Edges."""
        return convert.matrix_to_edges(item = self.contents)
       
    @abc.abstractproperty
    def matrix(self) -> Matrix:
        """Returns the stored graph as a Matrix."""
        return self.contents
       
    """ Class Methods """
    
    @abc.abstractclassmethod
    def from_adjacency(cls, item: Adjacency) -> Matrix:
        """Creates a Matrix instance from an Adjacency."""
        return cls(contents = convert.adjacency_to_matrix(item = item))
    
    @abc.abstractclassmethod
    def from_edges(cls, item: Edges) -> Matrix:
        """Creates a Matrix instance from an Edges."""
        return cls(contents = convert.edges_to_matrix(item = item))
    
    @abc.abstractclassmethod
    def from_matrix(cls, item: Matrix) -> Matrix:
        """Creates a Matrix instance from a Matrix."""
        return cls(contents = item[0], labels = item[1])
            
    """ Dunder Methods """
        
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        """Returns whether 'instance' meets criteria to be a subclass.

        Args:
            instance (object): item to test as an instance.

        Returns:
            bool: whether 'instance' meets criteria to be a subclass.
            
        """
        return check.is_matrix(item = instance)
    