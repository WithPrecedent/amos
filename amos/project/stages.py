"""
base:
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
from collections.abc import Mapping, MutableMapping, Set
import copy
import dataclasses
import itertools
from types import ModuleType
from typing import Any, ClassVar, Optional, Type, Union

from ..base import mappings
from ..core import composites
from ..core import graph
from ..repair import convert
from ..repair import modify
from ..report import recap

@dataclasses.dataclass
class Section(mappings.Dictionary):
    """Section of Outline with connections.

    Args:
        contents (MutableMapping[Hashable, Any]]): stored dictionary. Defaults 
            to an empty dict.
        default_factory (Any): default value to return when the 'get' method is 
            used. Defaults to None.
                          
    """
    contents: dict[str, Any] = dataclasses.field(default_factory = dict)
    default_factory: Any = None
    name: str = None

    """ Properties """
    
    @property
    def bases(self) -> dict[str, str]:
        return self._get_bases()
    
    @property
    def connections(self) -> dict[str, list[str]]:
        return self._get_connections()

    @property
    def designs(self) -> dict[str, str]:
        return self._get_designs()

    @property
    def nodes(self) -> list[str]:
        key_nodes = list(self.connections.keys())
        value_nodes = list(
            itertools.chain.from_iterable(self.connections.values()))
        return modify.deduplicate(item = key_nodes + value_nodes) 

    @property
    def other(self) -> dict[str, str]:
        return self._get_other()
    
    @property
    def suffixes(self) -> list[str]:
        return denovo.shared.library.subclasses.suffixes

    """ Public Methods """

    @classmethod
    def from_settings(cls, 
                      settings: amos.shared.bases.settings,
                      name: str,
                      **kwargs) -> Section:
        """[summary]

        Args:
            settings (amos.shared.bases.settings): [description]
            name (str):

        Returns:
            Section: derived from 'settings'.
            
        """        
        return cls(contents = settings[name], name = name, **kwargs)    
        
    """ Private Methods """

    def _get_bases(self) -> dict[str, str]:  
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        bases = {}
        for key in self.connections.keys():
            prefix, suffix = denovo.tools.divide_string(key)
            values = convert.iterify(self[key])
            if suffix.endswith('s'):
                base = suffix[:-1]
            else:
                base = suffix            
            bases.update(dict.fromkeys(values, base))
        return bases
         
    def _get_connections(self) -> dict[str, list[str]]:
        """[summary]

        Returns:
            dict[str, list[str]]: [description]
            
        """
        connections = {}
        keys = [k for k in self.keys() if k.endswith(self.suffixes)]
        for key in keys:
            prefix, suffix = denovo.tools.divide_string(key)
            values = convert.iterify(self[key])
            if prefix == suffix:
                if prefix in connections:
                    connections[self.name].extend(values)
                else:
                    connections[self.name] = values
            else:
                if prefix in connections:
                    connections[prefix].extend(values)
                else:
                    connections[prefix] = values
        return connections

    def _get_designs(self) -> dict[str, str]:  
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        designs = {}
        design_keys = [k for k in self.keys() if k.endswith('_design')]
        for key in design_keys:
            prefix, suffix = denovo.tools.divide_string(key)
            designs[prefix] = self[key]
        return designs
    
    def _get_other(self) -> dict[str, str]:
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        design_keys = [k for k in self.keys() if k.endswith('_design')]
        connection_keys = [k for k in self.keys() if k.endswith(self.suffixes)]
        exclude = design_keys + connection_keys
        return {k: v for k, v in self.contents.items() if k not in exclude}
        

@dataclasses.dataclass
class Outline(mappings.Dictionary):
    """Organized amos project settings with convenient accessors.

    Args:
        contents (denovo.configuration.TwoLevel): a two-level nested dict for 
            storing configuration options. Defaults to en empty dict.
        default (Any): default value to return when the 'get' method is used.
            Defaults to an empty dict.


    """
    contents: MutableMapping[str, Section] = dataclasses.field(
        default_factory = dict)
    default_factory: Any = None
    
    """ Properties """
    
    @property
    def bases(self) -> dict[str, str]:
        return self._get_bases()
    
    @property
    def connections(self) -> dict[str, list[str]]:
        return self._get_connections()

    @property
    def designs(self) -> dict[str, str]:
        return self._get_designs()

    @property 
    def initialization(self) -> dict[str, Any]:
        return self._get_initialization()  

    @property
    def nodes(self) -> list[str]:
        key_nodes = list(self.connections.keys())
        value_nodes = list(
            itertools.chain.from_iterable(self.connections.values()))
        return denovo.tools.deduplicate(item = key_nodes + value_nodes) 

    @property
    def other(self) -> dict[str, Any]:
        return self._get_other() 
    
    """ Public Methods """

    @classmethod
    def from_settings(cls, 
                      settings: amos.shared.bases.settings,
                      **kwargs) -> Outline:
        """[summary]

        Args:

        Returns:
            Outline: derived from 'settings'.
            
        """
        return amos.workshop.settings_to_outline(settings = settings, **kwargs)
             
    """ Private Methods """

    def _get_bases(self) -> dict[str, str]:  
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        bases = dict(zip(self.nodes, self.nodes))
        for section in self.values():
            bases.update(section.bases)
        return bases
      
    def _get_connections(self) -> dict[str, list[str]]:
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
    
    def _get_designs(self) -> dict[str, str]:  
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        designs = {}
        for section in self.values():
            designs.update(section.designs)
        return designs

    def _get_initialization(self) -> dict[str, dict[str, Any]]:  
        """[summary]

        Returns:
            dict[str, dict[str, Any]]: [description]
            
        """
        initialization = collections.defaultdict(dict)
        keys = [k.endswith('_parameters') for k in self.keys]
        for key in keys:
            prefix, suffix = denovo.tools.divide_string(key)
            initialization[prefix] = self[key]
        return initialization
    
    def _get_other(self) -> dict[str, str]:  
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        other = {}
        for section in self.values():
            other.update(section.other)
        return other


@dataclasses.dataclass
class Workflow(graph.System):
    """Project workflow implementation as a directed acyclic graph (DAG).
    
    Workflow stores its graph as an adjacency list. Despite being called an 
    "adjacency list," the typical and most efficient way to create one in python
    is using a dict. The keys of the dict are the nodes and the values are sets
    of the hashable summarys of other nodes.

    Workflow internally supports autovivification where a set is created as a 
    value for a missing key. 
    
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
    def cookbook(self) -> amos.base.Cookbook:
        """Returns the stored workflow as a Cookbook of Recipes."""
        return amos.workshop.workflow_to_cookbook(source = self)
            
    """ Dunder Methods """

    def __str__(self) -> str:
        """Returns prettier summary of the stored graph.

        Returns:
            str: a formatted str of class information and the contained 
                adjacency list.
            
        """
        return recap.beautify(item = self, package = 'amos')
