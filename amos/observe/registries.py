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
import collections
from collections.abc import Callable, MutableMapping, Sequence
import copy
import dataclasses
import functools
import inspect
from typing import Any, ClassVar, Optional, Type, Union

import miller

from . import containers
from . import utilities


""" Chained dict for Storing Subclasses and Instances """

@dataclasses.dataclass  # type: ignore
class Library(collections.ChainMap):
    """Stores classes and class instances.
    
    When searching for matches, instances are prioritized over classes.
    
    Args:
        classes (Catalog): a catalog of stored classes. Defaults to any empty
            Catalog.
        instances (Catalog): a catalog of stored class instances. Defaults to an
            empty Catalog.
        kinds (MutableMapping[str, set[str]]): a defaultdict with keys
            that are the different kinds of stored items and values which are
            sets of names of items that are of that kind. Defaults to an empty
            defaultdict which autovivifies sets as values.
    
    Attributes:
        maps (list[Catalog]): the ordered mappings to search, as required from
             inheriting from ChainMap.
                 
    """
    classes: containers.Catalog = containers.Catalog()
    instances: containers.Catalog = containers.Catalog()
    kinds: MutableMapping[str, set[str]] = dataclasses.field(
        default_factory = lambda: collections.defaultdict(set))

    """ Initialization Methods """
    
    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except AttributeError:
            pass
        # Sets order of Catalogs to search.
        self.maps = [self.instances, self.classes]
        
    """ Public Methods """
    
    def classify(self, item: Union[str, object, Type[Any]]) -> tuple[str, ...]:
        """Returns kind or kinds of 'item' based on 'kinds.'
        
        Args:
            item (Union[str, object, Type]): name of object or Type or an object
                or Type to be classified.
                
        Returns:
            tuple[str]: kinds of which 'item' is part of.
 
        """
        if not isinstance(item, str):
            item = miller.get_name(item = item)
        kinds = []  
        for kind, classes in self.kinds.items():  
            if item in classes:
                kinds.append(kind)
        return tuple(kinds)
       
    def deposit(
        self, 
        item: Union[Type[Any], object],
        kind: Optional[Union[str, Sequence[str]]] = None) -> None:
        """Adds 'item' to 'classes' and, possibly, 'instances'.

        If 'item' is a class, it is added to 'classes.' If it is an object, it
        is added to 'instances' and its class is added to 'classes'.
        
        Args:
            item (Union[Type, object]): class or instance to add to the Library
                instance.
            kind (Union[str, Sequence[str]]): kind(s) to add 'item'
                to. Defaults to None.
                
        """
        key = miller.get_name(item = item)
        base_key = None
        if inspect.isclass(item):
            self.classes[key] = item
        elif isinstance(item, object):
            self.instances[key] = item
            base = item.__class__
            base_key = miller.get_name(item = base)
            self.classes[base_key] = base
        else:
            raise TypeError(f'item must be a class or a class instance')
        if kind is not None:
            for classification in utilities.iterify(item = kind):
                self.kinds[classification].add(key)
                if base_key is not None:
                    self.kinds[classification].add(base_key)
        return
    
    def remove(self, name: str) -> None:
        """Removes an item from 'instances' or 'classes.'
        
        If 'name' is found in 'instances', it will not also be removed from 
        'classes'.

        Args:
            name (str): key name of item to remove.
            
        Raises:
            KeyError: if 'name' is neither found in 'instances' or 'classes'.

        """
        try:
            del self.instances[name]
        except KeyError:
            try:
                del self.classes[name]
            except KeyError:
                raise KeyError(f'{name} is not found in the library')
        return    

    def withdraw(
        self, 
        name: Union[str, Sequence[str]], 
        kwargs: Optional[Any] = None) -> Union[Type[Any], object]:
        """Returns instance or class of first match of 'name' from catalogs.
        
        The method prioritizes the 'instances' catalog over 'classes' and any
        passed names in the order they are listed.
        
        Args:
            name (Union[str, Sequence[str]]): key name(s) of stored item(s) 
                sought.
            kwargs (Mapping[Hashable, Any]): keyword arguments to pass to a
                newly created instance or, if the stored item is already an
                instance to be manually added as attributes.
            
        Raises:
            KeyError: if 'name' does not match a key to a stored item in either
                'instances' or 'classes'.
            
        Returns:
            Union[Type, object]: returns a class is 'kwargs' are None. 
                Otherwise, and object is returned.
            
        """
        names = utilities.iterify(name)
        item = None
        for key in names:
            for catalog in self.maps:
                try:
                    item = getattr(self, catalog)[key]
                    break
                except KeyError:
                    pass
            if item is not None:
                break
        if item is None:
            raise KeyError(f'No matching item for {name} was found')
        if kwargs is not None:
            if 'name' in item.__annotations__.keys() and 'name' not in kwargs:
                kwargs[name] = names[0]
            if inspect.isclass(item):
                item = item(**kwargs)
            else:
                for key, value in kwargs.items():
                    setattr(item, key, value)  
        return item # type: ignore


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
    namer: Callable[[Any], str] = miller.get_name
    
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
        # The default key for storing cls relies on the 'get_name' method, which
        # usually will use the snakecase name of 'item'.
        key = name or miller.get_name(item = cls)
        cls.registry[key] = item
        return