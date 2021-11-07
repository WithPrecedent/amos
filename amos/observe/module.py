"""
module: tools for python module introspection
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2021, Corey Rayburn Yung
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
   get_classes
   get_functions
   name_classes
   name_functions
   
   
ToDo:

"""
from __future__ import annotations
import importlib
import inspect
import sys
import types
from typing import Any, Optional, Type, Union

from ..repair import modify


""" Introspection Tools """
          
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
