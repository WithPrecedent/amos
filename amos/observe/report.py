"""
report: functions that return data stored in a passed python object
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
    Container Reporters:
        get_types
        get_types_dict
        get_types_list
        get_types_sequence
    Class and Instance Reporters:
        get_annotations
        get_attributes
        get_methods
        get_properties
        get_signatures
        get_variables
        name_attributes
        name_methods
        name_parameters
        name_properties
        name_variables
    Module Reporters:
        get_classes
        get_functions
        name_classes
        name_functions   
    File and Folder Reporters:
        get_file_paths
        get_folder_paths
        get_modules
        get_module_paths
        name_modules
        
ToDo:
    Add support for Kinds once that system is complete.

"""
from __future__ import annotations
from collections.abc import (
    Container, Hashable, Iterable, Mapping, MutableSequence, Sequence, Set)
import functools
import inspect
import pathlib
import sys
import types
from typing import Any, Optional, Type, Union

from . import check
from ..change import convert
from ..change import modify
from ..files import lazy


""" Container Reporters """   

@functools.singledispatch
def get_types(item: object) -> Optional[Union[
    tuple[Type[Any], ...], 
    tuple[tuple[Type[Any], ...], tuple[Type[Any], ...]]]]:
    """Returns types contained in 'item'.

    Args:
        item (object): item to examine.
    
    Returns:
        Optional[Union[tuple[Type[Any], ...], tuple[tuple[Type[Any], ...], 
            tuple[Type[Any], ...]]]]:: returns the types of things contained 
            in 'item'. Returns None if 'item' is not a container.
        
    """
    raise TypeError(f'item {item} is not supported by {__name__}')

@get_types.register(Mapping)  
def get_types_dict(
    item: Mapping[Hashable, Any]) -> Optional[
        tuple[tuple[Type[Any], ...], tuple[Type[Any], ...]]]:
    """Returns types contained in 'item'.

    Args:
        item (object): item to examine.
    
    Returns:
        Optional[tuple[Type[Any], ...]]: returns the types of things contained 
            in 'item'. Returns None if 'item' is not a container.
        
    """
    if isinstance(item, Mapping):
        key_types = get_types_sequence(item = item.keys())
        value_types = get_types_sequence(item = item.values())
        return tuple([key_types, value_types])
    else:
        return None

@get_types.register(MutableSequence)  
def get_types_list(item: list[Any]) -> Optional[tuple[Type[Any], ...]]:
    """Returns types contained in 'item'.

    Args:
        item (list[Any]): item to examine.
    
    Returns:
        Optional[tuple[Type[Any], ...]]: returns the types of things contained 
            in 'item'. Returns None if 'item' is not a container.
        
    """
    if isinstance(item, list):
        key_types = get_types_sequence(item = item.keys())
        value_types = get_types_sequence(item = item.values())
        return tuple([key_types, value_types])
    else:
        return None

@get_types.register(Sequence)    
def get_types_sequence(item: Sequence[Any]) -> Optional[tuple[Type[Any], ...]]:
    """Returns types contained in 'item'.

    Args:
        item (Sequence[Any]): item to examine.
    
    Returns:
        Optional[tuple[Type[Any], ...]]: returns the types of things contained 
            in 'item'. Returns None if 'item' is not a container.
        
    """
    if isinstance(item, Sequence):
        all_types = []
        for thing in item:
            kind = type(thing)
            if not kind in all_types:
                all_types.append(kind)
        return tuple(all_types)
    else:
        return None
    
""" Class and Instance Reporters """   
     
def get_annotations(
    item: object, 
    include_private: bool = False) -> dict[str, Type[Any]]:
    """Returns dict of attributes of 'item' with type annotations.
    
    Args:
        item (object): instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        dict[str, Any]: dict of attributes in 'item' (keys are attribute names 
            and values are type annotations) that are type annotated.
            
    """
    annotations = item.__annotations__
    if include_private:
        return annotations
    else:
        return {k: v for k, v in annotations.items() if not k.startswith('_')}

def get_attributes(
    item: object, 
    include_private: bool = False) -> dict[str, Any]:
    """Returns dict of attributes of 'item'.
    
    Args:
        item (Any): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        dict[str, Any]: dict of attributes in 'item' (keys are attribute names 
            and values are attribute values).
            
    """
    attributes = name_attributes(item = item, include_private = include_private)
    values = [getattr(item, m) for m in attributes]
    return dict(zip(attributes, values))

def get_methods(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> dict[str, types.MethodType]:
    """Returns dict of methods of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.

    Returns:
        dict[str, types.MethodType]: dict of methods in 'item' (keys are method 
            names and values are methods).
        
    """ 
    methods = name_methods(item = item, include_private = include_private)
    return [getattr(item, m) for m in methods]

def get_properties(
    item: object, 
    include_private: bool = False) -> dict[str, Any]:
    """Returns properties of 'item'.

    Args:
        item (object): instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.

    Returns:
        dict[str, Any]: dict of properties in 'item' (keys are property names 
            and values are property values).
        
    """    
    properties = name_properties(item = item, include_private = include_private)
    values = [getattr(item, p) for p in properties]
    return dict(zip(properties, values))

def get_signatures(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> dict[str, inspect.Signature]:
    """Returns dict of method signatures of 'item'.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.

    Returns:
        dict[str, inspect.Signature]: dict of method signatures in 'item' (keys 
            are method names and values are method signatures).
                   
    """ 
    methods = name_methods(item = item, include_private = include_private)
    signatures = [inspect.signature(getattr(item, m)) for m in methods]
    return dict(zip(methods, signatures))

def get_variables(
    item: object, 
    include_private: bool = False) -> dict[str, Any]:
    """Returns dict of attributes of 'item' that are not methods or properties.
    
    Args:
        item (object): instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        dict[str, Any]: dict of attributes in 'item' (keys are attribute names 
            and values are attribute values) that are not methods or properties.
            
    """
    attributes = name_attributes(item = item, include_private = include_private)
    methods = name_methods(item = item, include_private = include_private)
    properties = name_properties(item = item, include_private = include_private)
    variables = [
        a for a in attributes if a not in methods and a not in properties]
    values = [getattr(item, m) for m in variables]
    return dict(zip(variables, values))

def name_attributes(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> list[str]:
    """Returns attribute names of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        list[str]: names of attributes in 'item'.
            
    """
    names = dir(item)
    if not include_private:
        names = modify.drop_privates(item = names)
    return names

def name_methods(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> list[str]:
    """Returns method names of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        list[str]: names of methods in 'item'.
            
    """
    methods = [
        a for a in dir(item)
        if check.is_method(item = item, attribute = a)]
    if not include_private:
        methods = modify.drop_privates(item = methods)
    return methods

def name_parameters(item: Type[Any]) -> list[str]:
    """Returns list of parameters based on annotations of 'item'.

    Args:
        item (Type[Any]): class to get parameters to.

    Returns:
        list[str]: names of parameters in 'item'.
        
    """          
    return list(item.__annotations__.keys())

def name_properties(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> list[str]:
    """Returns method names of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        list[str]: names of properties in 'item'.
            
    """
    if not inspect.isclass(item):
        item = item.__class__
    properties = [
        a for a in dir(item)
        if check.is_property(item = item, attribute = a)]
    if not include_private:
        properties = modify.drop_privates(item = properties)
    return properties

def name_variables(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> list[str]:
    """Returns variable names of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        list[str]: names of attributes in 'item' that are neither methods nor
            properties.
            
    """
    names = [
        a for a in dir(item) if check.is_variable(item = item, attribute = a)]
    if not include_private:
        names = modify.drop_privates(item = names)
    return names

""" Module Reporters """

          
def get_classes(
    item: Union[types.ModuleType, str], 
    include_private: bool = False) -> list[Type[Any]]:
    """Returns list of classes in 'item'.
    
    Args:
        item (Union[types.ModuleType, str]): module or its name to inspect.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
        
    Returns:
        list[Type[Any]]: list of classes in 'item'.
        
    """
    if isinstance(item, str):
        item = sys.modules[item]
    classes = [
        m[1] for m in inspect.getmembers(item, inspect.isclass)
        if m[1].__module__ == item.__name__]
    if not include_private:
        classes = modify.drop_privates(item = classes)
    return classes
        
def get_functions(
    item: Union[types.ModuleType, str], 
    include_private: bool = False) -> list[types.FunctionType]:
    """Returns list of functions in 'item'.
    
    Args:
        item (Union[types.ModuleType, str]): module or its name to inspect.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
        
    Returns:
        list[Type[types.FunctionType]]: list of functions in 'item'.
        
    """
    if isinstance(item, str):
        item = sys.modules[item]
    functions = [
        m[1] for m in inspect.getmembers(item, inspect.isfunction)
        if m[1].__module__ == item.__name__]
    if not include_private:
        functions = modify.drop_privates(item = functions)
    return functions 
   
def name_classes(
    item: Union[types.ModuleType, str], 
    include_private: bool = False) -> list[str]:
    """Returns list of string names of classes in 'item'.
    
    Args:
        item (Union[types.ModuleType, str]): module or its name to inspect.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
        
    Returns:
        list[Type[types.FunctionType]]: list of functions in 'item'.
        
    """
    if isinstance(item, str):
        item = sys.modules[item]
    names = [    
        m[0] for m in inspect.getmembers(item, inspect.isclass)
        if m[1].__module__ == item.__name__]
    if not include_private:
        names = modify.drop_privates(item = names)
    return names
       
def name_functions(
    item: Union[types.ModuleType, str], 
    include_private: bool = False) -> list[str]:
    """Returns list of string names of functions in 'item'.
    
    Args:
        item (Union[types.ModuleType, str]): module or its name to inspect.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
        
    Returns:
        list[Type[types.FunctionType]]: list of functions in 'item'.
        
    """
    if isinstance(item, str):
        item = sys.modules[item]
    names = [
        m[0] for m in inspect.getmembers(item, inspect.isfunction)
        if m[1].__module__ == item.__name__]
    if not include_private:
        names = modify.drop_privates(item = names)
    return names

""" File and Folder Reporters """

def get_file_paths(
    item: Union[str, pathlib.Path],
    recursive: bool = False) -> list[pathlib.Path]:  
    """Returns list of non-python module file paths in 'item'.
    
    Args:
        item (Union[str, pathlib.Path]): path of folder to examine.
        recursive (bool): whether to include subfolders. Defaults to False.
        
    Returns:
        list[pathlib.Path]: a list of file paths in 'item'.
        
    """
    paths = get_paths(item = item, recursive = recursive)
    files = [p for p in paths if p.is_file()]
    return [f for f in files if f.is_file]

def get_folder_paths(
    item: Union[str, pathlib.Path],
    recursive: bool = False) -> list[pathlib.Path]:  
    """Returns list of folder paths in 'item'.
    
    Args:
        item (Union[str, pathlib.Path]): path of folder to examine.
        recursive (bool): whether to include subfolders. Defaults to False.
        
    Returns:
        list[pathlib.Path]: a list of folder paths in 'item'.
        
    """
    paths = get_paths(item = item, recursive = recursive)
    return [p for p in paths if check.is_folder(item = p)]

def get_modules(
    item: Union[str, pathlib.Path],
    recursive: bool = False) -> dict[types.ModuleType]:  
    """Returns dict of python module names and modules in 'item'.
    
    Args:
        item (Union[str, pathlib.Path]): path of folder to examine.
        recursive (bool): whether to include subfolders. Defaults to False.
        
    Returns:
        dict[str, types.ModuleType]: dict with str key names of python modules 
            and values as the corresponding modules.
        
    """
    return [
        lazy.from_file_path(path = p)
        for p in get_paths(item = item, recursive = recursive)]

def get_module_paths(
    item: Union[str, pathlib.Path],
    recursive: bool = False) -> list[pathlib.Path]:  
    """Returns list of python module paths in 'item'.
    
    Args:
        item (Union[str, pathlib.Path]): path of folder to examine.
        recursive (bool): whether to include subfolders. Defaults to False.
        
    Returns:
        list[pathlib.Path]: a list of python module paths in 'item'.
        
    """
    paths = get_paths(item = item, recursive = recursive)
    return [p for p in paths if check.is_module(item = p)]

def get_paths(
    item: Union[str, pathlib.Path], 
    suffix: str = '*',
    recursive: bool = False) -> list[pathlib.Path]:  
    """Returns list of all paths in 'item'.
    
    Args:
        item (Union[str, pathlib.Path]): path of folder to examine.
        suffix (str): file suffix to match. Defaults to '*' (all file suffixes). 
        recursive (bool): whether to include subfolders. Defaults to False.
        
    Returns:
        list[pathlib.Path]: a list of all paths in 'item'.
        
    """
    item = convert.pathlibify(item = item) 
    if recursive:
        return  list(folder.rglob(f'*.{suffix}')) # type: ignore
    else:
        return list(item.glob(f'*.{suffix}')) # type: ignore
      
def name_modules(
    item: Union[str, pathlib.Path],
    recursive: bool = False) -> list[str]:  
    """Returns list of python module names in 'item'.
    
    Args:
        item (Union[str, pathlib.Path]): path of folder to examine.
        recursive (bool): whether to include subfolders. Defaults to False.
        
    Returns:
        list[str]: a list of python module names in 'item'.
        
    """
    item = convert.pathlibify(item = item)
    kwargs = {'item': item, 'suffix': 'py', 'recursive': recursive}
    paths = [p.stem for p in get_paths(**kwargs)] # type: ignore
    return [str(p) for p in paths]
