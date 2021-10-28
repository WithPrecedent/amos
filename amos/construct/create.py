"""
create: factory creation tools
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
import abc
import ast
from collections.abc import Mapping
import dataclasses
from typing import Any, ClassVar, Optional, Type, Union

from ..repair import modify

 
@dataclasses.dataclass
class BaseFactory(abc.ABC):
    """Base class for instance factory mixins.

    Namespaces: create   
    
    """
    
    """ Required Subclass Methods """

    @abc.abstractclassmethod
    def create(cls, item: Any, *args: Any, **kwargs: Any) -> BaseFactory:
        """Implements technique for creating a (sub)class instance.

        Args:
            item (Any): argument indicating creation method to use.

        Returns:
            BaseFactory: instance of a SourcesFactory.
            
        """
        pass

          
@dataclasses.dataclass
class SourcesFactory(BaseFactory, abc.ABC):
    """Supports subclass creation using 'sources' class attribute.

    Args:
        sources (str, str]]): keys are str names of the types of the data 
            sources for object creation. For the appropriate creation 
            classmethod to be called, the types need to match the type of the
            first argument passed.
    
    Namespaces: create, sources, _get_create_method_name       
    
    """
    sources: ClassVar[Mapping[str, str]] = {}
    
    """ Public Methods """

    @classmethod
    def create(cls, item: Any, *args: Any, **kwargs: Any) -> SourcesFactory:
        """Calls corresponding creation class method to instance a class.
        
        For create to work properly, there should be a corresponding classmethod
        named f'from_{value in sources}'. If you would prefer a different naming
        format, you can subclass SourcesFactory and override the 
        '_get_create_method_name' classmethod.

        Raises:
            AttributeError: If an appropriate method does not exist for the
                data type of 'item.'
            KeyError: If the type of 'item' does not match a key in 
                'sources'.

        Returns:
            TypeFactory: instance of a SourcesFactory.
            
        """
        for kind, suffix in cls.sources.items():
            if isinstance(item, kind):
                method_name = cls._get_create_method_name(item = suffix)
                try:
                    method = getattr(cls, method_name)
                except AttributeError:
                    raise AttributeError(f'{method_name} does not exist')
                return method(item, *args, **kwargs)
        raise KeyError(
            f'item does not match any recognized types in sources attribute')

    """ Private Methods """
    
    @classmethod
    def _get_create_method_name(cls, item: str) -> str:
        """Returns classmethod name for creating an instance.
        
        Args:
            item (str): name corresponding to part of the str of the method
                name used for instancing.
                
        """
        return f'from_{item}'
          
          
@dataclasses.dataclass
class TypeFactory(BaseFactory, abc.ABC):
    """Supports subclass creation using str name of item type passed.

    Namespaces: create, _get_create_method_name       
    
    """
    
    """ Public Methods """

    @classmethod
    def create(cls, item: Any, *args: Any, **kwargs: Any) -> TypeFactory:
        """Calls construction method based on type of 'item'.
        
        For create to work properly, there should be a corresponding classmethod
        named f'from_{snake-case str name of type}'. If you would prefer a 
        different naming format, you can subclass TypeFactory and override the 
        '_get_create_method_name' classmethod.

        Raises:
            AttributeError: If an appropriate method does not exist for the
                data type of 'item.'

        Returns:
            TypeFactory: instance of a TypeFactory.
            
        """
        suffix = modify.snakify(item = str(type(item)))
        method_name = cls._get_create_method_name(item = suffix)
        try:
            method = getattr(cls, method_name)
        except AttributeError:
            raise AttributeError(f'{method_name} does not exist')
        return method(item, *args, **kwargs)

    """ Private Methods """
    
    @classmethod
    def _get_create_method_name(cls, item: str) -> str:
        """Returns classmethod name for creating an instance.
        
        Args:
            item (str): name corresponding to part of the str of the method
                name used for instancing.
                
        """
        return f'from_{item}'
      

@dataclasses.dataclass
class SubclassFactory(BaseFactory, abc.ABC):
    """Returns a subclass based on arguments passed to the 'create' method."""
        
    """ Public Methods """

    @classmethod
    def create(cls, item: str, *args: Any, **kwargs: Any) -> SubclassFactory:
        """Returns subclass based on 'item'
        
        A subclass in the '__subclasses__' attribute is selected based on the
        snake-case name of the subclass.
        
        Raises:
            KeyError: If a corresponding subclass does not exist for 'item.'

        Returns:
            SubclassFactory: instance of a SubclassFactory subclass.
            
        """
        options = {
            modify.snakify(item = s.__name__): s for s in cls.__subclasses__}
        try:
            return options[item](*args, **kwargs)
        except KeyError:
            raise KeyError(f'No subclass {item} was found')
 