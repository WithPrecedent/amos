"""
check: functions that check passed item and give a boolean result
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
    Contents Checkers:
        contains
        dict_contains
        list_contains
        set_contains
        tuple_contains
        parallel_contains
        serial_contains
    Simple Type Checkers:
        is_container: returns if an item is a container but not a str.
        is_function: returns if an item is a function type.
        is_iterable: returns if an item is iterable but not a str.
        is_nested: dispatcher which returns if an item is a nested container.
        is_nested_dict: returns if an item is a nested dict.
        is_nested_sequence: returns if an item is a nested sequence.
        is_nested_set: returns if an item is a nested set.
        is_sequence: returns if an item is a sequence but not a str.
    Composite Type Checkers:
        is_node: returns whether an item is a node.
        is_nodes: returns whether an item is a collection of nodes.
        is_edge: returns whether an item is an edge.
        is_composite: returns whether an item is a composite data structure.
        is_adjacency: returns whether an item is an adjacency list.
        is_matrix: returns whether an item is an adjacency matrix.
        is_edges: returns whether an item is a colleciton of edges.
        is_graph: returns whether an item is a graph.
        is_pipeline: returns whether an item is a pipeline of nodes.
        is_pipelines: returns whether an item is a collection of pipelines.
        is_tree: returns whether an item is a tree.
        is_forest: returns whether an item is a collection of trees.
    Attribute Checkers:
        has_attributes
        has_methods
        has_properties
        has_signatures
        has_traits
        is_class_attribute: returns whether an attribute is a class 
            attribute.
        is_method: returns whether an attribute of a class is a method. 
        is_property 
        is_variable: returns whether an attribute of a class is an
            ordinary data variable. 
    File and Folder Checkers:
        is_file
        is_folder
        is_module
        is_path   
    
To Do:
    Adding parsing functionality to commented signature functions to find
        equivalence when one signature has subtypes of the other signature
        (e.g., one type annotation is 'dict' and the other is 'MutableMapping').
        It might be necessary to create a separate Signature-like class to 
        implement this functionality. This includes fixing or abandoning 
        'has_annotations' due to issues matching type annotations.
    Add support for Kinds once that system is complete.
    Add support for types (using type annotations) in the 'contains' function so
        that 'contains' can be applied to classes and not just instances.
    Add 'dispatcher' framework to 'contains' once the dispatcher framework is
        completed in the 'bobbie' package and the Kind system is completed in
        the nagata package. This should replace existing usages of python's
        singledispatch, which doesn't propertly deal with subtypes.
    
"""
from __future__ import annotations
from collections.abc import (
    Collection, Container, Hashable, Iterable, Mapping, MutableMapping, 
    MutableSequence, Sequence, Set)
import functools
import inspect
import itertools
import pathlib
import types
from typing import Any, Optional, Type, Union
   
from ..change import convert 


""" Contents Checkers """
  
@functools.singledispatch
def contains(
    item: object,
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether 'item' contains the type(s) in 'contents'.

    Args:
        item (object): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.
        
    Raises:
        TypeError: if 'item' does not match any of the registered types.
        
    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    raise TypeError(f'item {item} is not supported by {__name__}')

@contains.register(Mapping)    
def dict_contains(
    item: Mapping[Hashable, Any], 
    contents: tuple[Union[Type[Any], tuple[Type[Any], ...]],
                    Union[Type[Any], tuple[Type[Any], ...]]]) -> bool:
    """Returns whether dict 'item' contains the type(s) in 'contents'.

    Args:
        item (Mapping[Hashable, Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return (
        serial_contains(item = item.keys(), contents = contents[0])
        and serial_contains(item = item.values(), contents = contents[1]))

@contains.register(MutableSequence)   
def list_contains(
    item: MutableSequence[Any],
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether list 'item' contains the type(s) in 'contents'.

    Args:
        item (MutableSequence[Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return serial_contains(item = item, contents = contents)

@contains.register(Set)   
def set_contains(
    item: Set[Any],
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether list 'item' contains the type(s) in 'contents'.

    Args:
        item (Set[Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return serial_contains(item = item, contents = contents)

@contains.register(tuple)   
def tuple_contains(
    item: tuple[Any, ...],
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether tuple 'item' contains the type(s) in 'contents'.

    Args:
        item (tuple[Any, ...]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    if isinstance(contents, tuple) and len(item) == len(contents):
        technique = parallel_contains
    else:
        technique = serial_contains
    return technique(item = item, contents = contents)

@contains.register(Sequence)   
def parallel_contains(
    item: Sequence[Any],
    contents: tuple[Type[Any], ...]) -> bool:
    """Returns whether parallel 'item' contains the type(s) in 'contents'.

    Args:
        item (Sequence[Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return all(isinstance(item[i], contents[i]) for i in enumerate(item))

@contains.register(Container)       
def serial_contains(
    item: Container[Any],
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether serial 'item' contains the type(s) in 'contents'.

    Args:
        item (Collection[Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return all(isinstance(i, contents) for i in item)

""" Simple Type Checkers """
    
def is_container(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is a container and not a str.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a container but not a str.
        
    """  
    if not inspect.isclass(item):
        item = item.__class__ 
    return issubclass(item, Container) and not issubclass(item, str)

def is_dict(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is a mutable mapping (generic dict type).
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a mutable mapping type.
        
    """  
    if not inspect.isclass(item):
        item = item.__class__ 
    return isinstance(item, MutableMapping) 

def is_function(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is a function type.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a function type.
        
    """  
    return isinstance(item, types.FunctionType)
  
def is_iterable(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is iterable and is NOT a str type.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is iterable but not a str.
        
    """ 
    if not inspect.isclass(item):
        item = item.__class__ 
    return issubclass(item, Iterable) and not issubclass(item, str)

def is_list(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is a mutable sequence (generic list type).
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a mutable list type.
        
    """
    if not inspect.isclass(item):
        item = item.__class__ 
    return isinstance(item, MutableSequence)

@functools.singledispatch
def is_nested(item: object) -> bool:
    """Returns if 'item' is nested at least one-level.
    
    Args:
        item (object): instance to examine.
        
    Raises:
        TypeError: if 'item' does not match any of the registered types.
        
    Returns:
        bool: if 'item' is a nested mapping.
        
    """ 
    raise TypeError(f'item {item} is not supported by {__name__}')

@is_nested.register(Mapping)   
def is_nested_dict(item: Mapping[Any, Any]) -> bool:
    """Returns if 'item' is nested at least one-level.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a nested mapping.
        
    """ 
    return (
        isinstance(item, Mapping) 
        and any(isinstance(v, Mapping) for v in item.values()))

@is_nested.register(Sequence)     
def is_nested_sequence(item: Sequence[Any]) -> bool:
    """Returns if 'item' is nested at least one-level.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a nested sequence.
        
    """ 
    return (
        is_sequence(item = item)
        and any(is_sequence(item = v) for v in item.values()))

@is_nested.register(Set)         
def is_nested_set(item: Set[Any]) -> bool:
    """Returns if 'item' is nested at least one-level.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a nested set.
        
    """ 
    return (
        is_set(item = item)
        and any(is_set(item = v) for v in item.values()))
        
def is_sequence(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is a sequence and is NOT a str type.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a sequence but not a str.
        
    """ 
    if not inspect.isclass(item):
        item = item.__class__ 
    return issubclass(item, Sequence) and not issubclass(item, str) 
        
def is_set(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is a Set (including generic type sets).
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a set.
        
    """ 
    if not inspect.isclass(item):
        item = item.__class__ 
    return issubclass(item, Set)

""" Composite Type Checkers """

def is_node(item: object) -> bool:
    """Returns whether 'item' is a node.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is a node.
    
    """
    if inspect.isclass(item):
        return hasattr(item, '__hash__') and hasattr(item, 'name')
    return isinstance(item, Hashable) and hasattr(item, 'name')

def is_nodes(item: object) -> bool:
    """Returns whether 'item' is a collection of nodes.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is a collection of nodes.
    
    """
    return isinstance(item, Collection) and all(is_node(item = i) for i in item)

def is_edge(item: object) -> bool:
    """Returns whether 'item' is an edge.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is an edge.
        
    """
    return (
        isinstance(item, tuple) 
        and len(item) == 2
        and is_node(item = item[0])
        and is_node(item = item[1]))

def is_composite(item: object) -> bool:
    """Returns whether 'item' is a collection of node connections.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is a collection of node connections.
        
    """
    return (
        is_adjacency(item = item)
        or is_edges(item = item)
        or is_graph(item = item)
        or is_matrix(item = item)
        or is_tree(item = item))

def is_adjacency(item: object) -> bool:
    """Returns whether 'item' is an adjacency list.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is an adjacency list.
        
    """
    if isinstance(item, MutableMapping):
        connections = list(item.values())
        nodes = list(itertools.chain(item.values()))
        return (
            all(isinstance(e, (Set)) for e in connections)
            and all(isinstance(n, (Set, Hashable)) for n in nodes))
    else:
        return False

def is_matrix(item: object) -> bool:
    """Returns whether 'item' is an adjacency matrix.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is an adjacency matrix.
        
    """
    if isinstance(item, tuple) and len(item) == 2:
        matrix = item[0]
        labels = item[1]
        connections = list(itertools.chain(matrix))
        return (
            isinstance(matrix, MutableSequence)
            and isinstance(labels, Sequence) 
            and not isinstance(labels, str)
            and all(isinstance(i, MutableSequence) for i in matrix)
            and all(isinstance(n, Hashable) for n in labels)
            and all(isinstance(e, int) for e in connections))
    else:
        return False

def is_edges(item: object) -> bool:
    """Returns whether 'item' is an edge list.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is an edge list.
    
    """
        
    return (
        isinstance(item, Sequence) 
        and not isinstance(item, str)
        and all(is_edge(item = i) for i in item))

def is_graph(item: object) -> bool:
    """Returns whether 'item' is a graph.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is a graph.
    
    """
        
    return (
        is_adjacency(item = item) 
        or is_matrix(item = item)
        or is_edges(item = item))

def is_pipeline(item: object) -> bool:
    """Returns whether 'item' is a pipeline.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is a pipeline.
    
    """
    return (
        isinstance(item, Sequence)
        and not isinstance(item, str)
        and all(is_node(item = i) for i in item))

def is_pipelines(item: object) -> bool:
    """Returns whether 'item' is a sequence of pipelines.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is a sequence of pipelines.
    
    """
    return (
        isinstance(item, Sequence)
        and not isinstance(item, str)
        and all(is_pipeline(item = i) for i in item)) 

def is_tree(item: object) -> bool:
    """Returns whether 'item' is a tree.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is a tree.
    
    """
    return (
        isinstance(item, MutableSequence)
        and all(isinstance(i, (MutableSequence, Hashable)) for i in item)) 
    
def is_forest(item: object) -> bool:
    """Returns whether 'item' is a dict of trees.

    Args:
        item (object): instance to test.

    Returns:
        bool: whether 'item' is a dict of trees.
    
    """
    return (
        isinstance(item, MutableMapping)
        and all(isinstance(i, Hashable) for i in item.keys())
        and all(is_tree(item = i) for i in item.values())) 

""" Attribute Checkers """

def has_attributes(
    item: Union[object, Type[Any]], 
    attributes: MutableSequence[str]) -> bool:
    """Returns whether 'attributes' exist in 'item'.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        attributes (MutableSequence[str]): names of attributes to check to see
            if they exist in 'item'.
            
    Returns:
        bool: whether all 'attributes' exist in 'items'.
    
    """
    return all(hasattr(item, a) for a in attributes)

def has_methods(
    item: Union[object, Type[Any]], 
    methods: Union[str, MutableSequence[str]]) -> bool:
    """Returns whether 'item' has 'methods' which are methods.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        methods (Union[str, MutableSequence[str]]): name(s) of methods to check 
            to see if they exist in 'item' and are types.MethodType.
            
    Returns:
        bool: whether all 'methods' exist in 'items' and are types.MethodType.
        
    """
    methods = list(convert.iterify(methods))
    return all(is_method(item = item, attribute = m) for m in methods)

def has_properties(
    item: Union[object, Type[Any]], 
    properties: Union[str, MutableSequence[str]]) -> bool:
    """Returns whether 'item' has 'properties' which are properties.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        properties (MutableSequence[str]): names of properties to check to see 
            if they exist in 'item' and are property type.
            
    Returns:
        bool: whether all 'properties' exist in 'items'.
        
    """
    properties = list(convert.iterify(properties))
    return all(is_property(item = item, attribute = p) for p in properties)
    
def has_signatures(
    item: Union[object, Type[Any]], 
    signatures: Mapping[str, inspect.Signature]) -> bool:
    """Returns whether 'item' has 'signatures' of its methods.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        signatures (Mapping[str, inspect.Signature]): keys are the names of 
            methods and values are the corresponding method signatures.
            
    Returns:
        bool: whether all 'signatures' exist in 'items'.
        
    """
    keys = [a for a in dir(item) if is_method(item = item, attribute = a)]
    values = [inspect.signature(getattr(item, m)) for m in keys]
    item_signatures = dict(zip(keys, values))
    pass_test = True
    for name, parameters in signatures.items():
        if (name not in item_signatures or item_signatures[name] != parameters):
            pass_test = False
    return pass_test
   
def has_traits(
    item: Union[object, Type[Any]],
    attributes: Optional[MutableSequence[str]] = None,
    methods: Optional[MutableSequence[str]] = None,
    properties: Optional[MutableSequence[str]] = None) -> bool:
    """Returns if 'item' has 'attributes', 'methods' and 'properties'.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        attributes (MutableSequence[str]): names of attributes to check to see
            if they exist in 'item'.
        methods (MutableSequence[str]): name(s) of methods to check to see if 
            they exist in 'item' and are types.MethodType.          
        properties (MutableSequence[str]): names of properties to check to see 
            if they exist in 'item' and are property type.
                          
    Returns:
        bool: whether all passed arguments exist in 'items'.    
    
    """
    if not inspect.isclass(item):
        item = item.__class__ 
    attributes = attributes or []
    methods = methods or []
    properties = properties or []
    signatures = signatures or {}
    return (
        has_attributes(item = item, attributes = attributes)
        and has_methods(item = item, methods = methods)
        and has_properties(item = item, properties = properties)
        and has_signatures(item = item, signatures = signatures))
 
def is_class_attribute(item: Union[object, Type[Any]], attribute: str) -> bool:
    """Returns if 'attribute' is a class attribute of 'item'."""
    if not inspect.isclass(item):
        item = item.__class__
    return (
        hasattr(item, attribute)
        and not is_method(item = item, attribute = attribute)
        and not is_property(item = item, attribute = attribute))
    
def is_container(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is a container and not a str.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a container but not a str.
        
    """  
    if not inspect.isclass(item):
        item = item.__class__ 
    return issubclass(item, Container) and not issubclass(item, str)
        
def is_method(item: Union[object, Type[Any]], attribute: Any) -> bool:
    """Returns if 'attribute' is a method of 'item'."""
    if isinstance(attribute, str):
        try:
            attribute = getattr(item, attribute)
        except AttributeError:
            return False
    return inspect.ismethod(attribute)
 
def is_property(item: Union[object, Type[Any]], attribute: Any) -> bool:
    """Returns if 'attribute' is a property of 'item'."""
    if not inspect.isclass(item):
        item = item.__class__
    if isinstance(attribute, str):
        try:
            attribute = getattr(item, attribute)
        except AttributeError:
            return False
    return isinstance(attribute, property)

def is_variable(item: Union[object, Type[Any]], attribute: str) -> bool:
    """Returns if 'attribute' is a simple data attribute of 'item'.

    Args:
        item (Union[object, Type[Any]]): [description]
        attribute (str): [description]

    Returns:
        bool: [description]
        
    """
    return (
        hasattr(item, attribute)
        and not is_function(item = item, attribute = attribute)
        and not is_property(item = item, attribute = attribute))

""" File and Folder Checkers """

def is_file(item: Union[str, pathlib.Path]) -> bool:
    """Returns whether 'item' is a non-python-module file.
    
    Args:
        item (Union[str, pathlib.Path]): path to check.
        
    Returns:
        bool: whether 'item' is a non-python-module file.
        
    """
    item = convert.pathlibify(item = item)
    return (
        item.exists() 
        and item.is_file() 
        and not item.suffix in ['.py', '.pyc'])

def is_folder(item: Union[str, pathlib.Path]) -> bool:
    """Returns whether 'item' is a path to a folder.
    
    Args:
        item (Union[str, pathlib.Path]): path to check.
        
    Returns:
        bool: whether 'item' is a path to a folder.
        
    """
    item = convert.pathlibify(item = item)
    return item.exists() and item.is_dir() # type: ignore

def is_module(item: Union[str, pathlib.Path]) -> bool:
    """Returns whether 'item' is a python-module file.
    
    Args:
        item (Union[str, pathlib.Path]): path to check.
        
    Returns:
        bool: whether 'item' is a python-module file.
        
    """
    item = convert.pathlibify(item = item)
    return item.exists() and item.is_file() and item.suffix in ['.py'] # type: ignore

def is_path(item: Union[str, pathlib.Path]) -> bool:
    """Returns whether 'item' is a currently existing path.
    
    Args:
        item (Union[str, pathlib.Path]): path to check.
        
    Returns:
        bool: whether 'item' is a currently existing path.
        
    """
    item = convert.pathlibify(item = item)
    return item.exists() # type: ignore


# def has_annotations(
#     item: Union[object, Type[Any]], 
#     attributes: Mapping[str, Type[Any]]) -> bool:
#     """Returns whether 'attributes' exist in 'item' and are the right type.
    
#     Args:
#         item (Union[object, Type[Any]]): class or instance to examine.
#         attributes (dict[str, Type[Any]]): dict where keys are the attribute 
#             names and values are the expected types of whose named attributes.
            
#     Returns
#         bool: whether all of the 'attributes' exist in 'item' and are of the
#             proper type.
            
#     """
#     matched = True
#     if inspect.isclass(item):
#         for attribute, value in attributes.items():
#             if value is not None:
#                 try:
#                     testing = getattr(item, attribute)
#                     testing = item.__annotations__[testing]
#                 except AttributeError:
#                     return False
#                 try:
#                     if not issubclass(testing, value):
#                         return False
#                 except TypeError:
#                     pass
#     else:
#         for attribute, value in attributes.items():
#             if value is not None:
#                 try:
#                     testing = getattr(item, attribute)
#                 except AttributeError:
#                     return False
#                 try:
#                     if not isinstance(testing, value):
#                         return False
#                 except TypeError:
#                     pass
#     return matched  
  