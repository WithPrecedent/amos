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
import collections.abc
import copy
import dataclasses
import itertools
from types import ModuleType
from typing import (Any, Callable, ClassVar, dict, Hashable, Iterable, list, 
                    Mapping, MutableMapping, MutableSequence, Optional, 
                    Sequence, Set, tuple, Type, Union)

import denovo
import more_itertools

import fiat


@dataclasses.dataclass
class Section(denovo.quirks.Factory, denovo.containers.Lexicon):
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
    sources: ClassVar[Mapping[Type, str]] = {
        fiat.shared.bases.settings : 'settings'}

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
        return denovo.tools.deduplicate(item = key_nodes + value_nodes) 

    @property
    def other(self) -> dict[str, str]:
        return self._get_other()
    
    @property
    def suffixes(self) -> list[str]:
        return denovo.shared.library.subclasses.suffixes

    """ Public Methods """

    @classmethod
    def from_settings(cls, 
                      settings: fiat.shared.bases.settings,
                      name: str,
                      **kwargs) -> Section:
        """[summary]

        Args:
            settings (fiat.shared.bases.settings): [description]
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
            values = denovo.tools.listify(self[key])
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
            values = denovo.tools.listify(self[key])
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
class Outline(denovo.quirks.Factory, denovo.containers.Lexicon):
    """Organized fiat project settings with convenient accessors.

    Args:
        contents (denovo.configuration.TwoLevel): a two-level nested dict for 
            storing configuration options. Defaults to en empty dict.
        default (Any): default value to return when the 'get' method is used.
            Defaults to an empty dict.
        default (Mapping[str, Mapping[str]]): any default options that should
            be used when a user does not provide the corresponding options in 
            their configuration settings. Defaults to an empty dict.
        infer_types (bool): whether values in 'contents' are converted to other 
            datatypes (True) or left alone (False). If 'contents' was imported 
            from an .ini file, all values will be strings. Defaults to True.

    """
    contents: MutableMapping[str, Section] = dataclasses.field(
        default_factory = dict)
    default_factory: Any = None
    sources: ClassVar[Mapping[Type, str]] = {
        fiat.shared.bases.settings : 'settings'}
    
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
                      settings: fiat.shared.bases.settings,
                      **kwargs) -> Outline:
        """[summary]

        Args:

        Returns:
            Outline: derived from 'settings'.
            
        """
        return fiat.workshop.settings_to_outline(settings = settings, **kwargs)
             
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
class Workflow(denovo.structures.System):
    """Project workflow implementation as a directed acyclic graph (DAG).
    
    Workflow stores its graph as an adjacency list. Despite being called an 
    "adjacency list," the typical and most efficient way to create one in python
    is using a dict. The keys of the dict are the nodes and the values are sets
    of the hashable summarys of other nodes.

    Workflow internally supports autovivification where a set is created as a 
    value for a missing key. 
    
    Args:
        contents (Adjacency): an adjacency list where the keys are nodes and the 
            values are sets of hash keys of the nodes which the keys are 
            connected to. Defaults to an empty a defaultdict described in 
            '_DefaultAdjacency'.
                  
    """  
    contents: denovo.structures.Adjacency = dataclasses.field(
        default_factory = lambda: collections.defaultdict(set))
    
    """ Properties """
    
    @property
    def cookbook(self) -> fiat.base.Cookbook:
        """Returns the stored workflow as a Cookbook of Recipes."""
        return fiat.workshop.workflow_to_cookbook(source = self)
            
    """ Dunder Methods """

    def __str__(self) -> str:
        """Returns prettier summary of the stored graph.

        Returns:
            str: a formatted str of class information and the contained 
                adjacency list.
            
        """
        return denovo.tools.beautify(item = self, package = 'fiat')
