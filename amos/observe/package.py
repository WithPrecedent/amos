"""
package: tools for package introspection
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
    get_file_paths
    get_folder_paths
    get_modules
    get_module_paths
    is_file
    is_folder
    is_module
    is_path
    name_modules
  
ToDo:


"""
from __future__ import annotations
import pathlib
import types
from typing import Any, Optional, Union

from . import utilities


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
    return [p for p in paths if is_folder(item = p)]

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
        utilities.from_file_path(path = p)
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
    return [p for p in paths if is_module(item = p)]

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
    item = utilities.pathlibify(item = item) 
    if recursive:
        return  list(folder.rglob(f'*.{suffix}')) # type: ignore
    else:
        return list(item.glob(f'*.{suffix}')) # type: ignore
    
def is_file(item: Union[str, pathlib.Path]) -> bool:
    """Returns whether 'item' is a non-python-module file.
    
    Args:
        item (Union[str, pathlib.Path]): path to check.
        
    Returns:
        bool: whether 'item' is a non-python-module file.
        
    """
    item = utilities.pathlibify(item = item)
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
    item = utilities.pathlibify(item = item)
    return item.exists() and item.is_dir() # type: ignore

def is_module(item: Union[str, pathlib.Path]) -> bool:
    """Returns whether 'item' is a python-module file.
    
    Args:
        item (Union[str, pathlib.Path]): path to check.
        
    Returns:
        bool: whether 'item' is a python-module file.
        
    """
    item = utilities.pathlibify(item = item)
    return item.exists() and item.is_file() and item.suffix in ['.py'] # type: ignore

def is_path(item: Union[str, pathlib.Path]) -> bool:
    """Returns whether 'item' is a currently existing path.
    
    Args:
        item (Union[str, pathlib.Path]): path to check.
        
    Returns:
        bool: whether 'item' is a currently existing path.
        
    """
    item = utilities.pathlibify(item = item)
    return item.exists() # type: ignore
      
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
    item = utilities.pathlibify(item = item)
    kwargs = {'item': item, 'suffix': 'py', 'recursive': recursive}
    paths = [p.stem for p in get_paths(**kwargs)] # type: ignore
    return [str(p) for p in paths]
