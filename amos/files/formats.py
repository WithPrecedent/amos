"""
template: file format class and helper functions
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
    FileFormat (factories.InstanceFactory): contains data needed for a Clerk-
        compatible file format.
    load_text:
    save_text:
    load_pickle:
    save_pickle:
    
ToDo:

    
"""
from __future__ import annotations
from collections.abc import Mapping, MutableSequence
import dataclasses
import pathlib
import sys
import types
from typing import Any, ClassVar, Optional, Type, Union

from ..containers import mappings
from ..builders import factories
from ..observe import report
from . import lazy
 
      
@dataclasses.dataclass
class FileFormat(factories.InstanceFactory):
    """File format information.

    Args:
        name (Optional[str]): the format name which should match the key when a 
            FileFormat instance is stored.
        module (Optional[str]): name of module where object to incorporate is, 
            which can either be a amos or non-amos module. Defaults to 
            None.
        extension (Optional[Union[str, MutableSequence[str]]]): str file 
            extension(s) to use. If more than one is listed, the first one is 
            used for saving new files and all will be used for loading. Defaults 
            to None.
        loader (Optional[Union[str, types.FunctionType]]): if a str, it is
            the name of import method in 'module' to use. Otherwise, it should
            be a function for loading. Defaults to None.
        saver (Optional[Union[str, types.FunctionType]]): if a str, it is
            the name of import method in 'module' to use. Otherwise, it should
            be a function for saved. Defaults to None.
        parameters (Mapping[str, str]]): shared parameters to use from 
            configuration settings where the key is the parameter name that the 
            load or save method should use and the value is the key for the 
            argument in the shared parameters. Defaults to an empty dict. 
        instances (ClassVar[mappings.Catalog]): project catalog of instances.
        
    """
    name: Optional[str] = None
    module: Optional[str] = None
    extension: Optional[Union[str, MutableSequence[str]]] = None
    loader: Optional[Union[str, types.FunctionType]] = None
    saver: Optional[Union[str, types.FunctionType]] = None
    parameters: Mapping[str, str] = dataclasses.field(default_factory = dict)
    instances: ClassVar[mappings.Catalog] = mappings.Catalog()
    
    """ Public Methods """
    
    def load(self, path: Union[str, pathlib.Path], **kwargs) -> Any:
        """[summary]

        Args:
            path (Union[str, pathlib.Path]): [description]

        Returns:
            Any: [description]
            
        """             
        method = self._validate_io_method(attribute = 'loader')
        return method(path, **kwargs)
    
    def save(self, item: Any, path: Union[str, pathlib.Path], **kwargs) -> None:
        """[summary]

        Args:
            item (Any): [description]
            path (Union[str, pathlib.Path]): [description]

        Returns:
            [type]: [description]
            
        """        
        method = self._validate_io_method(attribute = 'saver')
        method(item, path, **kwargs)
        return self
    
    """ Private Methods """
    
    def _validate_io_method(self, attribute: str) -> types.FunctionType:
        """[summary]

        Args:
            attribute (str): [description]

        Raises:
            AttributeError: [description]
            ValueError: [description]

        Returns:
            types.FunctionType: [description]
            
        """        
        value = getattr(self, attribute)
        if isinstance(value, str):
            if self.module is None:
                try:
                    method = getattr(self, value)
                except AttributeError:
                    try:
                        method = locals()[value]
                    except AttributeError:
                        raise AttributeError(
                            f'{value} {attribute} could not be found')
            else:
                method = lazy.from_import_path(
                    path = value, 
                    package = self.module)
            setattr(self, attribute, method)
        if not isinstance(value, types.FunctionType):
            raise ValueError(
                f'{attribute} must be a str, function, or method')
        return method
            
    """ Dunder Methods """
    
    @classmethod
    def __subclasshook__(cls, subclass: Type[Any]) -> bool:
        """Returns whether 'subclass' is a virtual or real subclass.

        Args:
            subclass (Type[Any]): item to test as a subclass.

        Returns:
            bool: whether 'subclass' is a real or virtual subclass.
            
        """
        return (
            subclass in cls.__subclasses__() 
            or (
                report.has_attributes(
                    item = subclass,
                    attributes = [
                        'name', 'module', 'extension', 'loader',
                        'saver', 'parameters'])
                and report.has_methods(
                    item = subclass,
                    methods = ['load', 'save'])))


""" Public Functions """

def load_text(path: Union[str, pathlib.Path], **kwargs) -> str:
    """[summary]

    Args:
        path (Union[str, pathlib.Path]): [description]

    Returns:
        str: [description]
        
    """    
    _file = open(path, 'r', **kwargs)
    loaded = _file.read()
    _file.close()
    return loaded

def save_text(item: Any, path: Union[str, pathlib.Path], **kwargs) -> None:
    """[summary]

    Args:
        item (Any): [description]
        path (Union[str, pathlib.Path]): [description]
        
    """    
    _file = open(path, 'w', **kwargs)
    _file.write(item)
    _file.close()
    return

def load_pickle(path: Union[str, pathlib.Path], **kwargs) -> str:
    """[summary]

    Args:
        path (Union[str, pathlib.Path]): [description]

    Returns:
        str: [description]
        
    """   
    _file = open(path, 'r', **kwargs)
    loaded = sys.modules['pickle'].load(_file)
    _file.close()
    return loaded

def save_pickle(
    item: Any, 
    path: Union[str, pathlib.Path], 
    **kwargs) -> None:
    """[summary]

    Args:
        item (Any): [description]
        path (Union[str, pathlib.Path]): [description]
        
    """   
    _file = open(path, 'w', **kwargs)
    sys.modules['pickle'].dump(item, _file)
    _file.close()
    return
