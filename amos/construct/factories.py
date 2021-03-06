"""
factories: factory creation tools
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
from collections.abc import Mapping, MutableMapping, Sequence
import copy
import dataclasses
from typing import Any, ClassVar, Optional, Type, Union

from ..base import mappings
from ..observe import registries
from ..observe import traits
from ..repair import modify

 
@dataclasses.dataclass
class BaseFactory(abc.ABC):
    """Base class for instance factory mixins."""
    
    """ Required Subclass Methods """

    @abc.abstractclassmethod
    def create(cls, item: Any, **kwargs: Any) -> BaseFactory:
        """Implements technique for creating a (sub)class instance.

        Args:
            item (Any): argument indicating creation method to use.

        Returns:
            BaseFactory: instance of a SourcesFactory.
            
        """
        pass

   
@dataclasses.dataclass
class InstanceFactory(BaseFactory):
    """Mixin which automatically registers instances.
    
    Args:
        instances (ClassVar[mappings.Catalog]): project catalog of instances.
            
    """
    instances: ClassVar[mappings.Catalog] = mappings.Catalog()
    
    """ Initialization Methods """
            
    def __post_init__(self) -> None:
        try:
            super().__post_init__(*args, **kwargs) # type: ignore
        except AttributeError:
            pass
        key = traits.get_name(item = self)
        self.__class__.instances[key] = self
        
    """ Public Methods """

    @classmethod
    def create(
        cls, 
        item: Union[str, Sequence[str]], 
        **kwargs: Any) -> InstanceFactory:
        """Creates an instance of a InstanceFactory subclass from 'item'.
        
        If kwargs are passed, they are added as attributes to the returned 
        instance.
        
        Args:
            item (Union[str, Sequence[str]]): key(s) for items stored in 
                'instances'.
                                
        Returns:
            InstanceFactory: an InstanceFactory instance created based on 'item' 
                and any passed arguments.
                
        """
        instance = copy.deepcopy(cls.instances[item])
        if kwargs:
            for key, value in kwargs.items():
                setattr(instance, key, value)
        return instance       
           
           
@dataclasses.dataclass
class LibraryFactory(BaseFactory):
    """Mixin which automatically registers subclasses and instances.
    
    Args:
        library (ClassVar[mappings.Library]): project library of classes, 
            instances, and base classes. 
            
    """
    library: ClassVar[mappings.Library] = mappings.Library()
    
    """ Initialization Methods """
    
    @classmethod
    def __init_subclass__(cls, *args: Any, **kwargs: Any):
        """Automatically registers subclass."""
        # Because LibraryFactory is used as a mixin, it is important to
        # call other base class '__init_subclass__' methods, if they exist.
        try:
            super().__init_subclass__(*args, **kwargs) # type: ignore
        except AttributeError:
            pass
        name = traits.get_name(item = cls)
        cls.library.deposit(item = cls, name = name)
            
    def __post_init__(self) -> None:
        try:
            super().__post_init__(*args, **kwargs) # type: ignore
        except AttributeError:
            pass
        key = traits.get_name(item = self)
        self.__class__.library.deposit(item = self, name = key)
    
    """ Public Methods """

    @classmethod
    def create(
        cls, 
        item: Union[str, Sequence[str]], 
        **kwargs: Any) -> LibraryFactory:
        """Creates an instance of a LibraryFactory subclass from 'item'.
        
        Args:
            item (Any): any supported data structure which acts as a source for
                creating a LibraryFactory or a str which matches a key in 
                'library'.
                                
        Returns:
            LibraryFactory: a LibraryFactory subclass instance created based 
                on 'item' and any passed arguments.
                
        """
        return cls.library.withdraw(item, **kwargs)

             
@dataclasses.dataclass
class RegistrarFactory(registries.Registrar, abc.ABC):
    """Mixin which automatically registers subclasses for use by a factory.
    
    Args:
        registry (ClassVar[MutableMapping[str, Type[Any]]]): key names are str
            names of a subclass (snake_case by default) and values are the 
            subclasses. Defaults to an empty dict.  
            
    """
    registry: ClassVar[MutableMapping[str, Type[Any]]] = {}
    
    """ Public Methods """

    @classmethod
    def create(cls, item: Any, **kwargs: Any) -> RegistrarFactory:
        """Creates an instance of a RegistrarFactory subclass from 'item'.
        
        Args:
            item (Any): any supported data structure which acts as a source for
                creating a RegistrarFactory or a str which matches a key in 
                'registry'.
                                
        Returns:
            RegistrarFactory: a RegistrarFactory subclass instance created based 
                on 'item' and any passed arguments.
                
        """
        if isinstance(item, str):
            try:
                return cls.registry[item](**kwargs)
            except KeyError:
                pass
        try:
            name = traits.get_name(item = item)
            return cls.registry[name](item, **kwargs)
        except KeyError:
            for name, kind in cls.registry.items():
                if (
                    abc.ABC not in kind.__bases__ 
                    and kind.__instancecheck__(instance = item)):
                    method = getattr(cls, f'from_{name}')
                    return method(item, **kwargs)       
            raise ValueError(
                f'Could not create {cls.__name__} from item because it '
                f'is not one of these supported types: '
                f'{str(list(cls.registry.keys()))}')
            
                      
@dataclasses.dataclass
class SourcesFactory(BaseFactory, abc.ABC):
    """Supports subclass creation using 'sources' class attribute.

    Args:
        sources (str, str]]): keys are str names of the types of the data 
            sources for object creation. For the appropriate creation 
            classmethod to be called, the types need to match the type of the
            first argument passed.  
    
    """
    sources: ClassVar[Mapping[str, str]] = {}
    
    """ Public Methods """

    @classmethod
    def create(cls, item: Any, **kwargs: Any) -> SourcesFactory:
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
                return method(item, **kwargs)
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
class SubclassFactory(BaseFactory, abc.ABC):
    """Returns a subclass based on arguments passed to the 'create' method."""
        
    """ Public Methods """

    @classmethod
    def create(cls, item: str, **kwargs: Any) -> SubclassFactory:
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
            return options[item](**kwargs)
        except KeyError:
            raise KeyError(f'No subclass {item} was found')
                 
          
@dataclasses.dataclass
class TypeFactory(BaseFactory, abc.ABC):
    """Supports subclass creation using str name of item type passed."""
    
    """ Public Methods """

    @classmethod
    def create(cls, item: Any, **kwargs: Any) -> TypeFactory:
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
        return method(item, **kwargs)

    """ Private Methods """
    
    @classmethod
    def _get_create_method_name(cls, item: str) -> str:
        """Returns classmethod name for creating an instance.
        
        Args:
            item (str): name corresponding to part of the str of the method
                name used for instancing.
                
        """
        return f'from_{item}'
