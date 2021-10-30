"""
workshop: helper classes for projects
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
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

"""
from __future__ import annotations
import abc
from collections.abc import (
    Hashable, Iterable, Iterator, Mapping, MutableMapping, Sequence)
import copy
import dataclasses
import inspect
import itertools
import pathlib
import types
from typing import Any, ClassVar, Optional, Type, TYPE_CHECKING, Union

from ..base import mappings
from ..construct import configuration
from ..observe import traits
from ..repair import convert
from ..repair import modify
from . import options

if TYPE_CHECKING:
    from . import interface


@dataclasses.dataclass
class ProjectSettings(configuration.Settings):
    """Loads and stores project configuration settings.

    To create settings instance, a user can pass as the 'contents' parameter a:
        1) pathlib file path of a compatible file type;
        2) string containing a a file path to a compatible file type;
                                or,
        3) 2-level nested dict.

    If 'contents' is imported from a file, settings creates a dict and can 
    convert the dict values to appropriate datatypes. Currently, supported file 
    types are: ini, json, toml, yaml, and python. If you want to use toml, yaml, 
    or json, the identically named packages must be available in your python
    environment.

    If 'infer_types' is set to True (the default option), str dict values are 
    automatically converted to appropriate datatypes (str, list, float, bool, 
    and int are currently supported). Type conversion is automatically disabled
    if the source file is a python module (assuming the user has properly set
    the types of the stored python dict).

    Because settings uses ConfigParser for .ini files, by default it stores 
    a 2-level dict. The desire for accessibility and simplicity amosted this 
    limitation. A greater number of levels can be achieved by having separate
    sections with names corresponding to the strings in the values of items in 
    other sections. 

    Args:
        contents (MutableMapping[Hashable, Any]): a dict for storing 
            configuration options. Defaults to en empty dict.
        default (Any): default value to return when the 'get' method is used.
            Defaults to an empty dict.
        default (Mapping[str, Mapping[str]]): any default options that should
            be used when a user does not provide the corresponding options in 
            their configuration settings. Defaults to an empty dict.
        infer_types (bool): whether values in 'contents' are converted to other 
            datatypes (True) or left alone (False). If 'contents' was imported 
            from an .ini file, all values will be strings. Defaults to True.

    """
    contents: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = dict)
    default_factory: Optional[Any] = dataclasses.field(
        default_factory = dict)
    default: Mapping[Hashable, Any] = dataclasses.field(
        default_factory = dict)
    infer_types: bool = True
    sources: ClassVar[Mapping[Type[Any], str]] = {
        MutableMapping: 'dictionary', 
        pathlib.Path: 'path',  
        str: 'path'}

    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Calls parent and/or mixin initialization method(s).
        try:
            super().__post_init__()
        except AttributeError:
            pass
        # Converts sections in 'contents' to Section types.
        self._sectionify()
        
    """" Private Methods """
    
    def _sectionify(self) -> None:
        new_contents = {}
        for key, value in self.contents.items():
            section = Section(contents = value, name = key)
            new_contents[key] = section
        self.contents = new_contents
        return


@dataclasses.dataclass
class Section(mappings.Dictionary):
    """Section of Outline with connections.

    Args:
        contents (MutableMapping[Hashable, Any]]): stored dictionary. Defaults 
            to an empty dict.
        default_factory (Any): default value to return when the 'get' method is 
            used. Defaults to None.
                          
    """
    contents: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = dict)
    default_factory: Optional[Any] = None
    name: Optional[str] = None

    """ Properties """
    
    @property
    def bases(self) -> dict[str, str]:
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        bases = {}
        for key in self.connections.keys():
            _, suffix = modify.cleave_str(key)
            values = convert.iterify(self[key])
            if suffix.endswith('s'):
                base = suffix[:-1]
            else:
                base = suffix            
            bases.update(dict.fromkeys(values, base))
        return bases
    
    @property
    def connections(self) -> dict[str, list[str]]:
        """[summary]

        Returns:
            dict[str, list[str]]: [description]
            
        """
        connections = {}
        keys = [k for k in self.keys() if k.endswith(self.suffixes)]
        for key in keys:
            prefix, suffix = modify.cleave_str(key)
            values = convert.iterify(self[key])
            if prefix == suffix:
                if prefix in connections:
                    connections[self.name].extend(values)
                else:
                    connections[self.name] = values
            else:
                if prefix in connections:
                    connections[prefix].extend(values)
                else:
                    connections[prefix] = values
        return connections

    @property
    def designs(self) -> dict[str, str]:
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        designs = {}
        design_keys = [k for k in self.keys() if k.endswith('_design')]
        for key in design_keys:
            prefix, _ = modify.cleave_str(key)
            designs[prefix] = self[key]
        return designs

    @property
    def nodes(self) -> list[str]:
        """[summary]

        Returns:
            list[str]: [description]
            
        """
        key_nodes = list(self.connections.keys())
        value_nodes = list(
            itertools.chain.from_iterable(self.connections.values()))
        return modify.deduplicate(item = key_nodes + value_nodes) 

    @property
    def other(self) -> dict[str, str]:
        """[summary]

        Returns:
            dict[str, str]: [description]
            
        """
        design_keys = [k for k in self.keys() if k.endswith('_design')]
        connection_keys = [k for k in self.keys() if k.endswith(self.suffixes)]
        exclude = design_keys + connection_keys
        return {k: v for k, v in self.contents.items() if k not in exclude}

    """ Public Methods """

    @classmethod
    def from_settings(
        cls, 
        settings: configuration.Settings,
        name: str,
        **kwargs) -> Section:
        """[summary]

        Args:
            settings (amos.shared.bases.settings): [description]
            name (str):

        Returns:
            Section: derived from 'settings'.
            
        """        
        return cls(contents = settings[name], name = name, **kwargs)    
      

@dataclasses.dataclass
class Director(Iterator):
    """Iterator for amos Project instances.
    
    
    """
    project: interface.Project = None
    converters: types.ModuleType = options.CONVERTERS
    
    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Calls parent and/or mixin initialization method(s).
        try:
            super().__post_init__()
        except AttributeError:
            pass
        # Sets index for iteration.
        self.index = 0
        # Validate stages
        self._validate_stages()
        
    """ Properties """
    
    @property
    def current(self) -> str:
        return list(self.stages.keys())[self.index]
    
    @property
    def previous(self) -> str:
        try:
            return list(self.stages.keys())[self.index -1]
        except IndexError:
            return None
          
    @property
    def stages(self) -> (
        Sequence[Union[str, Type[options.STAGE], options.STAGE]]):
        return self.project.stages
    
    @property
    def subsequent(self) -> str:
        try:
            return list(self.stages.keys())[self.index + 1]
        except IndexError:
            return None
       
    """ Public Methods """
    
    def advance(self) -> None:
        """Iterates through next stage."""
        return self.__next__()

    def complete(self) -> None:
        """Iterates through all stages."""
        for stage in self.stages:
            self.advance()
        return self    

    """ Private Methods """
    
    def _validate_stages(self) -> None:
        new_stages = []
        for stage in self.project.stages:
            new_stages.append(self._validate_stage(stage = stage))
        self.project.stages = new_stages
        return

    def _validate_stage(self, stage: Any) -> object:
        if isinstance(stage, str):
            try:
                stage = options.STAGE.create(stage)
            except KeyError:
                raise KeyError(f'{stage} was not found in Stage registry')
        if inspect.isclass(stage):
            stage = stage()
        return stage
            
        
    """ Dunder Methods """
    
    def __iter__(self) -> Iterable:
        """Returns iterable of 'stages'.
        
        Returns:
            Iterable: of 'stages'.
            
        """
        return self
 
    def __next__(self) -> None:
        """Completes a Stage instance."""
        if self.index < len(self.stages):
            source = self.previous or 'settings'
            product = self.stages[self.current]
            converter = getattr(self.converters, f'create_{product}')
            if self.project.settings['general']['verbose']:
                print(f'Creating {product} from {source}')
            kwargs = {'project': self.project}
            setattr(self.project, product, converter(**kwargs))
            self.index += 1
            if self.project.settings['general']['verbose']:
                print(f'Completed {product}')
        else:
            raise StopIteration
        return self


@dataclasses.dataclass
class ProjectLibrary(mappings.Library):
    """Stores project classes and class instances.
    
    When searching for matches, instances are prioritized over classes.
    
    Args:
        classes (Catalog): a catalog of stored classes. Defaults to any empty
            Catalog.
        instances (Catalog): a catalog of stored class instances. Defaults to an
            empty Catalog.
        bases (Catalog): base types of nodes. Defaults to an empty Catalog.
                 
    """
    classes: mappings.Catalog[str, Type[Any]] = dataclasses.field(
        default_factory = mappings.Catalog)
    instances: mappings.Catalog[str, object] = dataclasses.field(
        default_factory = mappings.Catalog)
    bases: mappings.Catalog[str, Type[Any]] = dataclasses.field(
        default_factory = mappings.Catalog)

    """ Public Methods """
    
    def classify(
        self,
        item: Union[str, Type[options.NODE], options.NODE]) -> str:
        """[summary]

        Args:
            item (str): [description]

        Returns:
            str: [description]
            
        """
        for name, _ in self.bases.items():
            if self.is_base(item = item, base = name):
                return name
        raise KeyError('item was not found in the project library')

    def instance(
        self, 
        name: Union[str, Sequence[str]], 
        *args, 
        **kwargs) -> object:
        """Returns instance of first match of 'name' in stored catalogs.
        
        The method prioritizes the 'instances' catalog over 'classes' and any
        passed names in the order they are listed.
        
        Args:
            name (Union[str, Sequence[str]]): [description]
            
        Raises:
            KeyError: [description]
            
        Returns:
            object: [description]
            
        """
        names = convert.iterify(name)
        primary = names[0]
        item = None
        for key in names:
            for catalog in ['instances', 'classes']:
                try:
                    item = getattr(self, catalog)[key]
                    break
                except KeyError:
                    pass
            if item is not None:
                break
        if item is None:
            raise KeyError(f'No matching item for {name} was found') 
        elif inspect.isclass(item):
            instance = item(primary, *args, **kwargs)
        else:
            instance = copy.deepcopy(item)
            for key, value in kwargs.items():
                setattr(instance, key, value)  
        return instance 

    def is_base(
        self, 
        item: Union[str, Type[options.NODE], options.NODE],
        base: str) -> bool:
        """[summary]

        Args:
            item (Union[str, Type[options.NODE], options.NODE]): [description]
            base (str): [description]

        Returns:
            bool: [description]
            
        """
        if isinstance(item, str):
            item = self[item]
        elif isinstance(item, options.NODE):
            item = item.__class__
        return issubclass(item, self.bases[base])        
        
    def parameterify(self, name: Union[str, Sequence[str]]) -> list[str]:
        """[summary]

        Args:
            name (Union[str, Sequence[str]]): [description]

        Returns:
            list[str]: [description]
            
        """        
        item = self.select(name = name)
        return list(item.__annotations__.keys())  


@dataclasses.dataclass
class LibraryFactory(abc.ABC):
    """Mixin which registers subclasses, instances, and bases.
    
    Args:
        library (ClassVar[ProjectLibrary]): project library of classes, 
            instances, and base classes. 
            
    """
    library: ClassVar[ProjectLibrary] = dataclasses.field(
        default_factory = ProjectLibrary)
    
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
        key = traits.get_name(item = cls)
        if abc.ABC in cls.__bases__:
            cls.library.bases[key] = cls
        else:
            cls.library.classes[key] = cls
            
    def __post_init__(self) -> None:
        try:
            super().__post_init__(*args, **kwargs) # type: ignore
        except AttributeError:
            pass
        key = traits.get_name(item = self)
        self.__class__.library[key] = self 
    
    """ Public Methods """

    @classmethod
    def create(
        cls, 
        item: Union[str, Sequence[str]], 
        *args: Any, 
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
        return cls.library.instance(item, *args, **kwargs)
        # if isinstance(item, str):
        #     try:
        #         return cls.library[item](*args, **kwargs)
        #     except KeyError:
        #         pass
        # try:
        #     name = traits.get_name(item = item)
        #     return cls.library[name](item, *args, **kwargs)
        # except KeyError:
        #     for name, kind in cls.library.items():
        #         if (
        #             abc.ABC not in kind.__bases__ 
        #             and kind.__instancecheck__(instance = item)):
        #             method = getattr(cls, f'from_{name}')
        #             return method(item, *args, **kwargs)       
        #     raise ValueError(
        #         f'Could not create {cls.__name__} from item because it '
        #         f'is not one of these supported types: '
        #         f'{str(list(cls.library.keys()))}')
            