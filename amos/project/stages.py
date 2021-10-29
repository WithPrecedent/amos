"""
stages: classes related to the different stages of an amos project
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
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

"""
from __future__ import annotations
import collections
from collections.abc import Hashable, Iterable, MutableMapping, Set
import dataclasses
import itertools
from typing import Any, ClassVar, Optional, Type, Union

from ..base import mappings
from ..core import composites
from ..core import graph
from ..core import hybrid
from ..repair import modify
from . import configuration
from . import workshop


@dataclasses.dataclass
class Stage(object):
    """Base classes for project stages.

    Args:
        contents (denovo.configuration.TwoLevel): a two-level nested dict for 
            storing configuration options. Defaults to en empty dict.

    """
    contents: Optional[Iterable] = None
    

@dataclasses.dataclass
class Outline(mappings.Dictionary, Stage):
    """Organized amos project settings with convenient accessors.

    Args:
        contents (MutableMapping[Hashable, Any]): stored dictionary. Defaults 
            to an empty dict.
        default_factory (Optional[Any]): default value to return or default 
            function to call when the 'get' method is used. Defaults to None. 

    """
    contents: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = dict)
    default_factory: Optional[Any] = None
    
    """ Properties """
    
    @property
    def bases(self) -> dict[str, str]:
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        bases = dict(zip(self.nodes, self.nodes))
        for section in self.values():
            bases.update(section.bases)
        return bases
    
    @property
    def connections(self) -> dict[str, list[str]]:
        """[summary]

        Returns:
            dict[str, list[str]]: [description]
            
        """
        connections = {}
        for section in self.values():
            for key, links in section.connections.items():
                if key in connections:
                    connections[key].extend(links)
                else:
                    connections[key] = links
        return connections

    @property
    def designs(self) -> dict[str, str]:
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        designs = {}
        for section in self.values():
            designs.update(section.designs)
        return designs

    @property 
    def initialization(self) -> dict[str, Any]:
        """[summary]

        Returns:
            dict[str, dict[str, Any]]: [description]
            
        """
        initialization = collections.defaultdict(dict)
        keys = [k.endswith('_parameters') for k in self.keys]
        for key in keys:
            prefix, suffix = modify.cleave_str(key)
            initialization[prefix] = self[key]
        return initialization

    @property
    def nodes(self) -> list[str]:
        """[summary]

        Returns:
            list[str]: [description]
            
        """
        key_nodes = list(self.connections.keys())
        value_nodes = list(
            itertools.chain.from_iterable(self.connections.values()))
        return modify.deduplicate(item = key_nodes + value_nodes) 

    @property
    def other(self) -> dict[str, Any]:
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        other = {}
        for section in self.values():
            other.update(section.other)
        return other
    
    """ Public Methods """

    @classmethod
    def from_settings(
        cls, 
        settings: configuration.Settings,
        **kwargs) -> Outline:
        """[summary]

        Args:

        Returns:
            Outline: derived from 'settings'.
            
        """
        return workshop.settings_to_outline(settings = settings, **kwargs)


@dataclasses.dataclass
class Workflow(graph.System, Stage):
    """Project workflow implementation as a directed acyclic graph (DAG).
    
    Workflow stores its graph as an adjacency list. Despite being called an 
    "adjacency list," the typical and most efficient way to create one in python
    is using a dict. The keys of the dict are the nodes and the values are sets
    of the hashable summarys of other nodes.
    
    Args:
        contents (MutableMapping[composites.Node, Set[composites.Node]]): keys 
            are nodes and values are sets of nodes (or hashable representations 
            of nodes). Defaults to a defaultdict that has a set for its value 
            format.
                  
    """  
    contents: MutableMapping[composites.Node, Set[composites.Node]] = (
        dataclasses.field(
            default_factory = lambda: collections.defaultdict(set)))
    
    """ Properties """
    
    @property
    def cookbook(self) -> hybrid.Pipelines:
        """Returns stored graph as pipelines."""
        return self.pipelines

