"""
workshop: functions for converting amos objects
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
from collections.abc import (
    Hashable, Iterable, Mapping, MutableMapping, MutableSequence, Sequence)
import dataclasses
from typing import Any, ClassVar, Optional, Type, Union

import more_itertools

from ..base import mappings
from ..repair import modify
from . import configuration
from . import stages


""" Configuration Parsing Functions """

def settings_to_outline(
    settings: configuration.Settings,**kwargs) -> stages.Outline:
    """[summary]

    Args:
        settings (amos.shared.bases.settings): [description]

    Returns:
        Outline: derived from 'settings'.
        
    """
    suffixes = denovo.shared.library.subclasses.suffixes
    outline = amos.Outline(**kwargs)
    section_base = amos.stages.Section
    for name, section in settings.items():
        if any(k.endswith(suffixes) for k in section.keys()):
            outline[name] = section_base.from_settings(settings = settings,
                                                       name = name)
    return outline
    
def create_workflow(project: amos.Project, **kwargs) -> amos.Workflow:
    """[summary]

    Args:
        project (amos.Project): [description]

    Returns:
        amos.Workflow: [description]
        
    """
    workflow = outline_to_workflow(outline = project.outline,
                                   library = project.library,
                                   **kwargs)
    return workflow

def outline_to_workflow(outline: amos.Outline, **kwargs) -> amos.Workflow:
    """[summary]

    Args:
        outline (amos.Outline): [description]
        library (denovo.containers.Library): [description]

    Returns:
        amos.Workflow: [description]
        
    """
    for name in outline.nodes:
        outline_to_component(name = name, outline = outline)
    workflow = amos.shared.bases.workflow
    workflow = outline_to_system(outline = outline, **kwargs)
    return workflow 
 
def outline_to_system(outline: amos.Outline) -> amos.Workflow:
    """[summary]

    Args:
        outline (amos.Outline): [description]
        library (nodes.Library, optional): [description]. Defaults to None.
        connections (dict[str, list[str]], optional): [description]. Defaults 
            to None.

    Returns:
        amos.structures.Graph: [description]
        
    """    
    connections = connections or outline_to_connections(
        outline = outline, 
        library = library)
    graph = amos.structures.Graph()
    for node in connections.keys():
        kind = library.classify(component = node)
        method = locals()[f'finalize_{kind}']
        graph = method(
            node = node, 
            connections = connections,
            library = library, 
            graph = graph)     
    return graph

def outline_to_component(name: str, 
                         outline: amos.Outline, 
                         **kwargs) -> amos.base.Component:
    """[summary]

    Args:
        name (str): [description]
        section (str): [description]
        outline (amos.Outline): [description]
        library (nodes.Library, optional): [description]. Defaults to None.
        connections (dict[str, list[str]], optional): [description]. Defaults 
            to None.
        design (str, optional): [description]. Defaults to None.
        recursive (bool, optional): [description]. Defaults to True.
        overwrite (bool, optional): [description]. Defaults to False.

    Returns:
        nodes.Component: [description]
    
    """
    design = outline.designs[name] or None
    base = outline.bases[name]
    initialization = outline_to_initialization(
        name = name, 
        design = design,
        section = section, 
        outline = outline,
        library = library)
    initialization.update(kwargs)
    if 'parameters' not in initialization:
        initialization['parameters'] = outline_to_implementation(
            name = name, 
            design = design,
            outline = outline)
    component = library.instance(name = [name, design], **initialization)
    return component

def outline_to_initialization(
    name: str, 
    section: str,
    design: str,
    outline: amos.Outline,
    library: nodes.Library) -> dict[Hashable, Any]:
    """Gets parameters for a specific Component from 'outline'.

    Args:
        name (str): [description]
        section (str): [description]
        design (str): [description]
        outline (amos.Outline): [description]
        library (nodes.Library): [description]

    Returns:
        dict[Hashable, Any]: [description]
        
    """
    suboutline = outline[section]
    parameters = library.parameterify(name = [name, design])
    possible = tuple(i for i in parameters if i not in ['name', 'contents'])
    parameter_keys = [k for k in suboutline.keys() if k.endswith(possible)]
    kwargs = {}
    for key in parameter_keys:
        prefix, suffix = amos.tools.divide_string(key)
        if key.startswith(name) or (name == section and prefix == suffix):
            kwargs[suffix] = suboutline[key]
    return kwargs  
        
def outline_to_implementation(
    name: str, 
    design: str,
    outline: amos.Outline) -> dict[Hashable, Any]:
    """[summary]

    Args:
        name (str): [description]
        design (str): [description]
        outline (amos.Outline): [description]

    Returns:
        dict[Hashable, Any]: [description]
        
    """
    try:
        parameters = outline[f'{name}_parameters']
    except KeyError:
        try:
            parameters = outline[f'{design}_parameters']
        except KeyError:
            parameters = {}
    return parameters

def finalize_serial(
    node: str,
    connections: dict[str, list[str]],
    library: nodes.Library,
    graph: amos.structures.Graph) -> amos.structures.Graph:
    """[summary]

    Args:
        node (str): [description]
        connections (dict[str, list[str]]): [description]
        library (nodes.Library): [description]
        graph (amos.structures.Graph): [description]

    Returns:
        amos.structures.Graph: [description]
        
    """    
    connections = _serial_order(
        name = node, 
        connections = connections)
    nodes = list(more_itertools.collapse(connections))
    if nodes:
        graph.extend(nodes = nodes)
    return graph      

def _serial_order(
    name: str,
    connections: dict[str, list[str]]) -> list[Hashable]:
    """[summary]

    Args:
        name (str): [description]
        directive (core.Directive): [description]

    Returns:
        list[Hashable]: [description]
        
    """   
    organized = []
    components = connections[name]
    for item in components:
        organized.append(item)
        if item in connections:
            organized_connections = []
            connections = _serial_order(
                name = item, 
                connections = connections)
            organized_connections.append(connections)
            if len(organized_connections) == 1:
                organized.append(organized_connections[0])
            else:
                organized.append(organized_connections)
    return organized   


""" Workflow Executing Functions """

def workflow_to_summary(project: amos.Project, **kwargs) -> amos.Project:
    """[summary]

    Args:
        project (amos.Project): [description]

    Returns:
        nodes.Component: [description]
        
    """
    # summary = None
    # print('test workflow', project.workflow)
    # print('test paths', project.workflow.paths)
    # print('test parser contents', library.instances['parser'].contents)
    # print('test parser paths', library.instances['parser'].paths)
    summary = configuration.SUMMARY()
    print('test project paths', project.workflow.paths)
    # for path in enumerate(project.workflow.paths):
    #     name = f'{summary.prefix}_{i + 1}'
    #     summary.add({name: workflow_to_result(
    #         path = path,
    #         project = project,
    #         data = project.data)})
    return summary
        
def workflow_to_result(
    path: Sequence[str],
    project: amos.Project,
    data: Any = None,
    library: nodes.Library = None,
    result: core.Result = None,
    **kwargs) -> object:
    """[summary]

    Args:
        name (str): [description]
        path (Sequence[str]): [description]
        project (amos.Project): [description]
        data (Any, optional): [description]. Defaults to None.
        library (nodes.Library, optional): [description]. Defaults to None.
        result (core.Result, optional): [description]. Defaults to None.

    Returns:
        object: [description]
        
    """    
    library = library or configuration.LIBRARY
    result = result or configuration.RESULT
    data = data or project.data
    result = result()
    for node in path:
        print('test node in path', node)
        try:
            component = library.instance(name = node)
            result.add(component.execute(project = project, **kwargs))
        except (KeyError, AttributeError):
            pass
    return result
