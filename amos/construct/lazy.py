"""
lazy: lazy importing utilities
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

    
ToDo:


"""
from __future__ import annotations
from collections.abc import MutableMapping
import dataclasses
import importlib
import importlib.util
import pathlib
import sys
import types
from typing import Any, ClassVar, Optional, Union


""" Importing Tools """

def from_path(
    path: Union[str, pathlib.Path], 
    name: Optional[str] = None) -> Any:
    """Imports and returns module from import or file path at 'name'.

    Args:
        path (Union[pathlib.Path, str]): import or file path of module to load.
        name (Optional[str]): name to store module at in 'sys.modules'. If it
            is None, the stem of 'path' is used. Defaults to None.
            
    Returns:
        Any:
        
    """
    if isinstance(path, pathlib.Path) or '\\' in path or '\/' in path:
        return from_file_path(path = path, name = name)
    else:
        return from_import_path(path = path)

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
    if isinstance(path, str):
        path = pathlib.Path(path)
    if name is None:
        name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f'Failed to create spec from {path}')
    else:
        return importlib.util.module_from_spec(spec)
    
def from_import_path(path: str, package: Optional[str] = None) -> Any:
    """[summary]

    Args:
        path (str): [description]
        package (Optional[str], optional): [description]. Defaults to None.

    Returns:
        Any: [description]
        
    """
    try:
        return sys.modules[path]
    except KeyError:
        if package is None:
            kwargs = {}
        else:
            kwargs = {'package': package}
        techniques = [
            absolute_import,
            absolute_supackage_import,
            relative_import,
            relative_subpackage_import]
        for technique in techniques:
            try:
                return technique(path, **kwargs)
            except ModuleNotFoundError:
                item = path.split('.')[-1]
                module_name = path[:-len(item) - 1]
                module = technique(module_name, **kwargs)
                return getattr(module, item)  
        raise ModuleNotFoundError(f'{path} could not be imported') 
             
def absolute_import(path: str) -> Any:
    """[summary]

    Args:
        path (str): [description]

    Returns:
        Any: [description]
        
    """
    if path.startswith('.'):
        path = path[1:]
        return relative_subpackage_import(path = path)
    return importlib.import_module(path)
 
def absolute_supackage_import(path: str, package: str) -> Any:
    """[summary]

    Args:
        path (str): [description]
        package (str): [description]. 

    Returns:
        Any: [description]
        
    """
    if path.startswith('.'):
        path = path[1:]
        return relative_subpackage_import(path = path, package = package)
    return importlib.import_module(path, package = package)

def relative_import(path: str) -> Any:
    """[summary]

    Args:
        path (str): [description]

    Returns:
        Any: [description]
        
    """
    if not path.startswith('.'):
        path = '.' + path
    return importlib.import_module(sys.path_importer_cache)

def relative_subpackage_import(path: str, package: str) -> Any:
    """[summary]

    Args:
        path (str): [description]
        package (str): [description]. 

    Returns:
        Any: [description]
        
    """
    if not path.startswith('.'):
        path = '.' + path
    return importlib.import_module(path, package = package)

def from_importables(
    name: str, 
    importables: MutableMapping[str, str],
    package: Optional[str] = None) -> Any:
    """Lazily imports modules and items within them.
    
    Lazy importing means that modules are only imported when they are first
    accessed. This can save memory and keep namespaces decluttered.
    
    This code is adapted from PEP 562: https://www.python.org/dev/peps/pep-0562/
    which outlines how the decision to incorporate '__getattr__' functions to 
    modules allows lazy loading. Rather than place this function solely within
    '__getattr__', it is included here seprately so that it can easily be called 
    by '__init__.py' files throughout amos and by users (as 
    'amos.load.from_dict').
    
    To effectively use 'from_dict' in an '__init__.py' file, the user needs to 
    pass a 'importables' dict which indicates how users should accesss imported 
    modules and included items. This modules includes an example 'importables' 
    dict and how to easily add this function to a '__getattr__' function.
    
    Instead of limiting its lazy imports to full import paths as the example in 
    PEP 562, this function has 2 major advantages:
        1) It allows importing items within modules and not just modules. The
            function first tries to import 'name' assuming it is a module. But 
            if that fails, it parses the last portion of 'name' and attempts to 
            import the preceding module and then returns the item within it.
        2) It allows import paths that are less than the full import path by
            using the 'importables' dict. 'importables' has keys which are the 
            name of the attribute being sought and values which are the full 
            import path (dropping the leading '.'). 'importables' thus acts as 
            the normal import block in an __init__.py file but insures that all 
            importing is done lazily.
            
    Args:
        name (str): name of module or item within a module.
        package (str): name of package from which the module is sought.
        importables (MutableMapping[str, str]): keys are the access names for 
            items sought and values are the import path where the item is 
            actually located.
        
    Raises:
        AttributeError: if there is no module or item matching 'name' im 
            'importables'.
        
    Returns:
        Any: a module or item stored within a module.
        
    """
    if name in importables:
        return from_import_path(path = importables[name], package = package)
    else:
        raise KeyError(f'{name} is not in importables') 


""" Class Implementations """

@dataclasses.dataclass
class Importer(object):
    """Lazy importer that converts a dict to loadable .
    
    """
    package: str
    importables: MutableMapping[str, str] = dataclasses.field(
        default_factory = dict)
    registry: ClassVar[MutableMapping[str, Importer]] = {}
    
    """ Initialization Methods """
    
    def __post_init__(self) -> None:
        # Stores instance in the class registry.
        self.__class__.registry[self.package] = self
        
    """ Public Methods """
    
    def load(self, name: str) -> Any:
        if name in self.importables:
            if not isinstance(self.importables[name], str):
                return self.importables[name]
            else:
                imported = from_importables(
                    name = name, 
                    importables = self.importables,
                    package = self.package)
                self.importables[name] = imported
                return imported
        else:
            raise KeyError(f'{name} is not in importables')
           

@dataclasses.dataclass
class Delayed(object):
    """Mixin that converts str attributes to imported items when accessed.
    
    Only str values of attributes with a '.' in them are assumed to be import
    paths.
    
    After an item is imported, it is assigned to the formerly str attribute so
    that it does not need to be reloaded.
    
    """
 
    """ Dunder Methods """
    
    def __getattribute__(self, name: str) -> Any:
        item = super().__getattribute__(name)
        if isinstance(item, str) and '.' in item:
            imported = from_import_path(item = name)
            super().__setattr__(name, imported)
            return imported
        else:
            return item
                
