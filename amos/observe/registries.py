"""
registries: classes and functions for registration
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
    Library (ChainMap): registry for storing, accessing, and instancing classes
        and instances.
        
"""
from __future__ import annotations
import abc
from collections.abc import Callable, MutableMapping, Sequence
import copy
import dataclasses
import functools
import inspect
from typing import Any, ClassVar, Optional, Type, Union

from . import traits


""" Basic Registration System """

@dataclasses.dataclass
class registered(object):
    """
    
    registered violates the normal python convention of naming classes in 
    capital case because it is only designed to be used as a callable decorator, 
    where lowercase names are the norm.
 
    Args:

    
    """
    wrapped: Callable[..., Optional[Any]]
    defaults: dict[str, Callable[..., Optional[Any]]] = dataclasses.field(
        default_factory = dict)
    namer: Callable[[Any], str] = traits.get_name
    
    """ Initialization Methods """
        
    def __call__(
        self, 
        *args: Any, 
        **kwargs: Any) -> Callable[..., Optional[Any]]:
        """Allows class to be called as a decorator.
        
        Returns:
            Callable[..., Optional[Any]]: callable after it has been registered.
        
        """
        # Updates 'wrapped' for proper introspection and traceback.
        functools.update_wrapper(self, self.wrapped)
        # Copies key attributes and functions to wrapped item.
        self.wrapped.register = self.register
        self.wrapped.registry = self.__class__.registry
        if inspect.isclass(self.wrapped):
            self.wrapped.__init_subclass__ = Registrar.__init_subclass__
        return self.wrapped(*args, **kwargs)        

    """ Properties """
    
    @property
    def registry(self) -> MutableMapping[str, Type[Any]]:
        """Returns internal 'kinds' registry with builtin python types added.
        
        Returns:
            MutableMapping[str, Type[Any]]: dict of str keys and values of Kind 
                subclasses and builtin python types.
                
        """
        if self.defaults:
            complete = copy.deepcopy(self._registry)
            complete.update(self.defaults)
            return complete 
        else:
            return self._registry
    
    """ Public Methods """
    
    @classmethod
    def register(cls, item: Type[Any], name: Optional[str] = None) -> None:
        """Adds 'item' to 'registry'.
        
        """
        # The default key for storing cls is its snakecase name.
        key = name or cls.namer(cls)
        cls.registry[key] = item
        return


@dataclasses.dataclass
class Registrar(object):
    """Mixin which automatically registers subclasses.
    
    Args:
        registry (ClassVar[MutableMapping[str, Type[Any]]]): key names are str
            names of a subclass (snake_case by default) and values are the 
            subclasses. Defaults to an empty dict.  
            
    """
    registry: ClassVar[MutableMapping[str, Type[Any]]] = {}
    
    """ Initialization Methods """
    
    @classmethod
    def __init_subclass__(cls, *args: Any, **kwargs: Any):
        """Automatically registers subclass in 'registry'."""
        # Because Registrar will often be used as a mixin, it is important to
        # call other base class '__init_subclass__' methods, if they exist.
        try:
            super().__init_subclass__(*args, **kwargs) # type: ignore
        except AttributeError:
            pass
        cls.register(item = cls)

    """ Public Methods """
    
    @classmethod
    def register(cls, item: Type[Any], name: Optional[str] = None) -> None:
        """Adds 'item' to 'registry'.
        
        A separate 'register' method is included so that virtual subclasses can
        also be registered.
        
        Args:
            item (Type[Any]): a class to add to the registry.
            name (Optional[str]): name to use as the key when 'item' is stored
                in 'registry'. Defaults to None. If not passed, the 'get_name'
                method will be used to 
        
        """
        # if abc.ABC not in cls.__bases__:
        # The default key for storing cls relies on the 'get_name' method, 
        # which usually will use the snakecase name of 'item'.
        key = name or traits.get_name(item = cls)
        cls.registry[key] = item
        return   
