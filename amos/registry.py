"""
registries: classes and functions for registration
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
    registered (object): a decorator for automatic registration of wrapped 
        classes and functions.
    Registrar (object): automatically stores subclasses in 'registry' class 
        attribute and allows virtual subclass registration using the 'register'
        class method.
        
"""
from __future__ import annotations
from collections.abc import Callable, MutableMapping, Sequence
import copy
import dataclasses
import functools
import inspect
from typing import Any, ClassVar, Optional, Type, Union

from . import convert


""" Basic Registration System """

@dataclasses.dataclass
class registered(object):
    """Decorator that automatically registers wrapped class or function.
    
    registered violates the normal python convention of naming classes in 
    capital case because it is only designed to be used as a callable decorator, 
    where lowercase names are the norm.
    
    All registered functions and classes are stored in the 'registry' class 
    attribute of the wrapped item (even if it is a function). So, it is 
    accessible with '{wrapped item name}.registry'. If the wrapped item is a 
    class is subclassed, those subclasses will be registered as well via the 
    '__init_subclass__' method which is copied from the Registrar class.
        
    Wrapped functions and classes are automatically added to the stored registry
    with the 'namer' function. Virtual subclasses can be added using the
    'register' method which is automatically added to the wrapped function or
    class.
 
    Args:
        wrapped (Callable[..., Optional[Any]]): class or function to be stored.
        default (dict[str, Callable[..., Optional[Any]]]): any items to include
             in the registry without requiring additional registration. Defaults
             to an empty dict.
        namer (Callable[[Any], str]): function to infer key names of wrapped
            functions and classes. Defaults to the 'namify' function in amos.
    
    """
    wrapped: Callable[..., Optional[Any]]
    defaults: dict[str, Callable[..., Optional[Any]]] = dataclasses.field(
        default_factory = dict)
    namer: Callable[[Any], str] = convert.namify
    
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
        """Returns internal registry.
        
        Returns:
            MutableMapping[str, Type[Any]]: dict of str keys and values of
                registered items.
                
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
                in 'registry'. Defaults to None. If not passed, the 'namify'
                method will be used to 
        
        """
        # if abc.ABC not in cls.__bases__:
        # The default key for storing cls relies on the 'namify' method, 
        # which usually will use the snakecase name of 'item'.
        key = name or convert.namify(item = cls)
        cls.registry[key] = item
        return   
