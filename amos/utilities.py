"""
utilities: shared helper functions
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
     
"""
from __future__ import annotations
from collections.abc import Hashable, Iterable, MutableMapping
import importlib
import pathlib
import re
import types
from typing import Any, Callable, Optional, Type, Union


def drop_dunders(item: list[Any]) -> list[Any]:
    """Drops items in 'item' with names beginning with an underscore.

    Args:
        item (list[Any]): attributes, methods, and properties of a class.

    Returns:
        list[Any]: attributes, methods, and properties that do not start with an
            underscore.
        
    """
    if len(item) > 0 and hasattr(item[0], '__name__'):
        return [
            i for i in item 
            if not i.__name__.startswith('__') 
            and not i.__name__.endswith('__')]
    else:
        return [
            i for i in item if not i.startswith('__') and not i.endswith('__')]
    
def drop_privates(item: list[Any]) -> list[Any]:
    """Drops items in 'item' with names beginning with an underscore.

    Args:
        item (list[Any]): attributes, methods, and properties of a class.

    Returns:
        list[Any]: attributes, methods, and properties that do not start with an
            underscore.
        
    """
    if len(item) > 0 and hasattr(item[0], '__name__'):
        return [i for i in item if not i.__name__.startswith('_')]
    else:
        return [i for i in item if not i.startswith('_')]

def from_file_path(
    path: Union[pathlib.Path, str], 
    name: Optional[str] = None) -> types.ModuleType:
    """Imports and returns module from file path at 'name'.

    Args:
        path (Union[pathlib.Path, str]): file path of module to load.
        name (Optional[str]): name to store module at in 'sys.modules'. If it
            is None, the stem of 'path' is used. Defaults to None.
    Returns:
        types.ModuleType: imported module.
        
    """
    path = pathlibify(item = path)
    if name is None:
        name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f'Failed to create spec from {path}')
    else:
        imported = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported)
        return imported

def get_name(item: Any, default: Optional[str] = None) -> Optional[str]:
    """Returns str name representation of 'item'.
    
    Args:
        item (Any): item to determine a str name.
        default(Optional[str]): default name to return if other methods at name
            creation fail.

    Returns:
        str: a name representation of 'item.'
        
    """        
    if isinstance(item, str):
        return item
    else:
        if hasattr(item, 'name') and isinstance(item.name, str):
            return item.name
        else:
            try:
                return snakify(item.__name__) # type: ignore
            except AttributeError:
                if item.__class__.__name__ is not None:
                    return snakify( # type: ignore
                        item.__class__.__name__) 
                else:
                    return default
                  
def iterify(item: Any) -> Iterable:
    """Returns 'item' as an iterable, but does not iterate str types.
    
    Args:
        item (Any): item to turn into an iterable

    Returns:
        Iterable: of 'item'. A str type will be stored as a single item in an
            Iterable wrapper.
        
    """     
    if item is None:
        return iter(())
    elif isinstance(item, (str, bytes)):
        return iter([item])
    else:
        try:
            return iter(item)
        except TypeError:
            return iter((item,))
        
def pathlibify(item: Union[str, pathlib.Path]) -> pathlib.Path:
    """Converts string 'path' to pathlib.Path object.

    Args:
        item (Union[str, pathlib.Path]): either a string summary of a
            path or a pathlib.Path object.

    Returns:
        pathlib.Path object.

    Raises:
        TypeError if 'path' is neither a str or pathlib.Path type.

    """
    if isinstance(item, str):
        return pathlib.Path(item)
    elif isinstance(item, pathlib.Path):
        return item
    else:
        raise TypeError('item must be str or pathlib.Path type')
    
def snakify(item: str) -> str:
    """Converts a capitalized str to snake case.

    Args:
        item (str): str to convert.

    Returns:
        str: 'item' converted to snake case.

    """
    item = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', item)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', item).lower()    
   