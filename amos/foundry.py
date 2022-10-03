"""
foundry: easy-to-use factory mixins
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
    BaseFactory (ABC): base class for amos factory mixins. It requires 
        subclasses have a 'create' classmethod.
    InstanceFactory (BaseFactory): mixin that stores all subclass instances in 
        the 'instances' class attribute and returns stored instances when the 
        'create' classmethod is called.
    LibraryFactory (BaseFactory): mixin that stores all subclasses and 
        subclass instances in the 'library' class attribute and returns stored 
        subclasses and/or instances when the 'create' classmethod is called.
    SourceFactory (BaseFactory): mixin that calls the appropriate creation 
        method based on the type of passed first argument to 'create' and the
        types stored in the keys of the 'sources' class attribute.
    StealthFactory (BaseFactory): mixin that returns stored subclasses when the 
        'create' classmethod is called without having a 'subclasses' class 
        attribute like SubclassFactory.
    SubclassFactory (BaseFactory): mixin that stores all subclasses in the 
        'subclasses' class attribute and returns stored subclasses when the 
        'create' classmethod is called.
    TypeFactory (BaseFactory): mixin that calls the appropriate creation 
        method based on the type of passed first argument to 'create' and the
        snakecase name of the type. This factory is prone to significant 
        key errors unless you are sure of the snakecase names of all possible 
        submitted type names. SourceFactory avoids this problem by allowing you
        to declare corresponding types and string names.
            
ToDo:
    Determine if there is any value to commented out RegistrarFactory class

"""

from __future__ import annotations
import abc
import contextlib
from collections.abc import Mapping, MutableMapping, Sequence
import copy
import dataclasses
from typing import Any, ClassVar, Optional, Type, Union

from . import convert
from . import mapping
from . import modify

 
@dataclasses.dataclass
class BaseFactory(abc.ABC):
    """Base class for factory mixins."""
    
    """ Required Subclass Methods """

    @abc.abstractclassmethod
    def create(
        cls,
        source: Any, 
        **kwargs: Any) -> Union[Type[BaseFactory], BaseFactory]:
        """Returns a subclass or subclass instance.

        Args:
            source (Any): argument indicating creation method to use.

        Returns:
            BaseFactory: instance of a SourceFactory.
            
        """
        pass

   
@dataclasses.dataclass
class InstanceFactory(BaseFactory):
    """Mixin which automatically registers and stores subclass instances.
    
    Args:
        instances (ClassVar[mapping.Catalog]): catalog of subclass instances.
            
    """
    instances: ClassVar[mapping.Catalog] = mapping.Catalog()
    
    """ Initialization Methods """
            
    def __post_init__(self) -> None:
        """Automatically registers subclass."""
        # Because InstanceFactory is used as a mixin, it is important to
        # call other base class '__init_subclass__' methods, if they exist.
        with contextlib.suppress(AttributeError):
            super().__post_init__(*args, **kwargs) # type: ignore
        key = convert.namify(item = self)
        self.__class__.instances[key] = self
        
    """ Public Methods """

    @classmethod
    def create(
        cls, 
        source: Union[str, Sequence[str]], 
        **kwargs: Any) -> Union[InstanceFactory, list[InstanceFactory]]:
        """Creates an instance of a InstanceFactory subclass from 'source'.
        
        If kwargs are passed, they are added as attributes to the returned 
        instance.
        
        Args:
            source (str): key(s) for item(s) stored in 'instances'.
            
        Returns:
            InstanceFactory: an InstanceFactory instance created based on 
                'source' and any passed arguments.
                
        """
        if isinstance(source, list):
            instances = []
            for key in source:
                instances.append(cls._create_instance(source = key, **kwargs))
            return instances
        elif isinstance(cls.instances[source], list):
            pass
        else:
            return cls._create_instance(source = source, **kwargs)  
    
    """ Private Methods """
    
    @classmethod
    def _create_instance(cls, source: str, **kwargs: Any) -> InstanceFactory:
        """Creates an instance of a InstanceFactory subclass from 'source'.
        
        If kwargs are passed, they are added as attributes to the returned 
        instance.
        
        Args:
            source (str): key for item stored in 'instances'.
            
        Returns:
            InstanceFactory: an InstanceFactory instance created based on 
                'source' and any passed arguments.
                
        """
        instance = copy.deepcopy(cls.instances[source])
        if kwargs:
            for key, value in kwargs.items():
                setattr(instance, key, value)
        return instance   


@dataclasses.dataclass
class LibraryFactory(BaseFactory):
    """Mixin which automatically registers and stores subclasses and instances.
    
    When the 'create' method is called on this class, a matching subclass is
    sought first. If no matching subclass is found, subclass instances are 
    searched. In either case, an instance is returned. If kwargs are passed,
    they are used to either initialize a subclass or added to an instance as
    attributes.
    
    Args:
        library (ClassVar[mapping.Library]): library of subclasses and 
            instances. 
            
    """
    library: ClassVar[mapping.Library] = mapping.Library()
    
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
        name = convert.namify(item = cls)
        cls.library.deposit(item = cls, name = name)
            
    def __post_init__(self) -> None:
        with contextlib.suppress(AttributeError):
            super().__post_init__(*args, **kwargs) # type: ignore
        key = convert.namify(item = self)
        self.__class__.library.deposit(item = self, name = key)
    
    """ Public Methods """

    @classmethod
    def create(
        cls, 
        source: str, 
        **kwargs: Any) -> LibraryFactory:
        """Creates an instance of a LibraryFactory subclass from 'source'.
        
        Args:
            source (Any): any supported data structure which acts as a source 
                for creating a LibraryFactory or a str which matches a key in 
                'library'.
                                
        Returns:
            LibraryFactory: a LibraryFactory subclass instance created based 
                on 'source' and any passed arguments.
                
        """
        return cls.library.withdraw(source, **kwargs)

                   
@dataclasses.dataclass
class SourceFactory(BaseFactory, abc.ABC):
    """Mixin that returns subclasses using 'sources' class attribute.

    Unlike typical factories, this one does not require an additional class 
    attribute to be added to store registered subclasses. Instead, it requires
    subclasses to add creation methods using a common naming format.
    
    This factory acts as a dispatcher to call other methods based on the type
    passed. Unlike TypeFactory, SourceFactory is more forgiving by allowing the
    type passed to a subtype of the type listed as a key in the 'sources' class
    attribute.
    
    Args:
        sources (Type, str]]): keys are types of the data sources for object 
            creation. Values are the corresponding str name of the type which
            should have a class method called 'from_{str name of type}'. Because
            the 'create' method will call the first method for which 'source' 
            matches a key, you should put specific types before general types in
            the 'sources' dict.
    
    """
    sources: ClassVar[Mapping[Type, str]] = {}
    
    """ Public Methods """

    @classmethod
    def create(cls, source: Any, **kwargs: Any) -> SourceFactory:
        """Calls corresponding creation class method to instance a class.
        
        For create to work properly, there should be a corresponding classmethod
        named f'from_{value in sources}'. If you would prefer a different naming
        format, you can subclass SourceFactory and override the 
        '_get_create_method_name' classmethod.

        Raises:
            AttributeError: If an appropriate method does not exist for the
                data type of 'source.'
            KeyError: If the type of 'source' does not match a key in 
                'sources'.

        Returns:
            TypeFactory: instance of a SourceFactory.
            
        """
        for kind, suffix in cls.sources.items():
            if isinstance(source, kind):
                method_name = cls._get_create_method_name(item = suffix)
                try:
                    method = getattr(cls, method_name)
                except AttributeError:
                    raise AttributeError(f'{method_name} does not exist')
                return method(source, **kwargs)
        raise KeyError(
            f'source does not match any recognized types in sources attribute')

    """ Private Methods """
    
    @classmethod
    def _get_create_method_name(cls, source: str) -> str:
        """Returns classmethod name for creating an instance.
        
        Args:
            source (str): name corresponding to part of the str of the method
                name used for instancing.
                
        """
        return f'from_{source}'
      

@dataclasses.dataclass
class StealthFactory(BaseFactory):
    """Mixin that returns a subclass without requiring a new attribute.
    
    Unlike typical factories, this one does not require an additional class 
    attribute to be added to store registered subclasses. Instead, it relies on
    pre-existing data and dynamically adds keys to create a dict facade.
    
    This factory uses the subclasses stored in '__subclasses__' dunder attribute
    that is automatically created with every class. It creates a dict on the fly
    with key names being snakecase of the stored subclasses '__name__' 
    attributes.
    
    """
        
    """ Public Methods """

    @classmethod
    def create(cls, source: str) -> StealthFactory:
        """Returns subclass based on 'source'
        
        A subclass in the '__subclasses__' attribute is selected based on the
        snake-case name of the subclass.
        
        Raises:
            KeyError: If a corresponding subclass does not exist for 'source.'

        Returns:
            StealthFactory: a StealthFactory subclass.
            
        """
        options = {
            modify.snakify(item = s.__name__): s for s in cls.__subclasses__}
        try:
            return options[source]
        except KeyError:
            raise KeyError(f'No subclass {source} was found')
        

@dataclasses.dataclass
class SubclassFactory(BaseFactory):
    """Mixin which automatically registers and stores subclasses.
    
    The reason this is a factory and not a mere registry is that it returns a
    deepcopy of a stored subclass so that any modifications to a retrieved
    subclass are not applied to future retrieved subclasses.
    
    Args:
        subclasses (ClassVar[mapping.Catalog]): project catalog of subclasses.
            
    """
    subclasses: ClassVar[mapping.Catalog] = mapping.Catalog()
    
    """ Initialization Methods """
    
    @classmethod
    def __init_subclass__(cls, *args: Any, **kwargs: Any):
        """Automatically registers subclass."""
        # Because SubclassFactory is used as a mixin, it is important to
        # call other base class '__init_subclass__' methods, if they exist.
        with contextlib.suppress(AttributeError):
            super().__init_subclass__(*args, **kwargs) # type: ignore
        name = convert.namify(item = cls)
        cls.subclasses[name] = cls
    
    """ Public Methods """

    @classmethod
    def create(cls, source: str) -> Type[SubclassFactory]:
        """Returns a copy of a subclass of SubclassFactory based on 'source'.
        
        Args:
            source (str): key for item stored in 'subclasses'.
                                
        Returns:
            SubclassFactory: a copy of a SubclassFactory subclass created based 
                on 'source'.
                
        """
        return copy.deepcopy(cls.subclasses[source])
                    
                            
@dataclasses.dataclass
class TypeFactory(BaseFactory, abc.ABC):
    """Mixin that returns subclass using the type or str name of the type.

    Unlike typical factories, this one does not require an additional class 
    attribute to be added to store registered subclasses. Instead, it requires
    subclasses to add creation methods using a common naming format.
       
    This factory acts as a dispatcher to call other methods based on the type
    or name of the type passed. By default, using the '_get_create_method_name'
    method, the format for such methods should be 'from_{str name of type}'.
    
    """
    
    """ Public Methods """

    @classmethod
    def create(cls, source: Any, **kwargs: Any) -> TypeFactory:
        """Calls construction method based on type of 'source'.
        
        For create to work properly, there should be a corresponding classmethod
        named f'from_{snake-case str name of type}'. If you would prefer a 
        different naming format, you can subclass TypeFactory and override the 
        '_get_create_method_name' classmethod.

        Raises:
            AttributeError: If an appropriate method does not exist for the
                data type of 'source.'

        Returns:
            TypeFactory: instance of a TypeFactory.
            
        """
        suffix = modify.snakify(item = str(type(source)))
        method_name = cls._get_create_method_name(item = suffix)
        try:
            method = getattr(cls, method_name)
        except AttributeError:
            raise AttributeError(f'{method_name} does not exist')
        return method(source, **kwargs)

    """ Private Methods """
    
    @classmethod
    def _get_create_method_name(cls, item: str) -> str:
        """Returns classmethod name for creating an instance.
        
        Args:
            source (str): name corresponding to part of the str of the method
                name used for instancing.
                
        """
        return f'from_{item}'

             
# @dataclasses.dataclass
# class RegistrarFactory(registries.Registrar, abc.ABC):
#     """Mixin which automatically registers subclasses for use by a factory.
    
#     Args:
#         registry (ClassVar[MutableMapping[str, Type[Any]]]): key names are str
#             names of a subclass (snake_case by default) and values are the 
#             subclasses. Defaults to an empty dict.  
            
#     """
#     registry: ClassVar[MutableMapping[str, Type[Any]]] = {}
    
#     """ Public Methods """

#     @classmethod
#     def create(cls, item: Any, **kwargs: Any) -> RegistrarFactory:
#         """Creates an instance of a RegistrarFactory subclass from 'item'.
        
#         Args:
#             item (Any): any supported data structure which acts as a source for
#                 creating a RegistrarFactory or a str which matches a key in 
#                 'registry'.
                                
#         Returns:
#             RegistrarFactory: a RegistrarFactory subclass instance created based 
#                 on 'item' and any passed arguments.
                
#         """
#         if isinstance(item, str):
#             try:
#                 return cls.registry[item](**kwargs)
#             except KeyError:
#                 pass
#         try:
#             name = trait.namify(item = item)
#             return cls.registry[name](item, **kwargs)
#         except KeyError:
#             for name, kind in cls.registry.items():
#                 if (
#                     abc.ABC not in kind.__bases__ 
#                     and kind.__instancecheck__(instance = item)):
#                     method = getattr(cls, f'from_{name}')
#                     return method(item, **kwargs)       
#             raise ValueError(
#                 f'Could not create {cls.__name__} from item because it '
#                 f'is not one of these supported types: '
#                 f'{str(list(cls.registry.keys()))}')
            
   