"""
interface: primary access point and interface for an amos project
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
    Project

"""
from __future__ import annotations
from collections.abc import (
    Hashable, Iterable, Iterator, MutableMapping, MutableSequence, Sequence)
import dataclasses
import inspect
import pathlib
from typing import Any, ClassVar, Optional, Type, Union
import warnings

import more_itertools

from ..core import composites
from . import configuration
from . import filing
from . import helpers
from . import stages


@dataclasses.dataclass
class Project(composites.Node):
    """Interface for an amos project.
    
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout amos. For example, if an amos 
            instance needs outline from an Outline instance, 'name' should 
            match the appropriate section name in an Outline instance. Defaults 
            to None. 
        outline (OutlineSources): an Outline-compatible subclass or instance, 
            a str or pathlib.Path containing the file path where a file of a 
            supported file type with outline for an Outline instance is 
            located, or a 2-level mapping containing outline. Defaults to the 
            default Outline instance.
        clerk (ClerkSources): a Clerk-compatible class or a str or pathlib.Path 
            containing the full path of where the root folder should be located 
            for file input and output. A 'clerk' must contain all file path and 
            import/export methods for use throughout amos. Defaults to the 
            default Clerk instance.    
        stages (ClassVar[Sequence[Union[str, core.Stage]]]): a list of Stages or 
            strings corresponding to keys in 'core.library'. Defaults to a list 
            of strings listed in the dataclass field.
        data (object): any data object for the project to be applied. If it is 
            None, an instance will still execute its workflow, but it won't
            apply it to any external data. Defaults to None.
        identification (str): a unique identification name for an amos Project. 
            The name is used for creating file folders related to the project. 
            If it is None, a str will be created from 'name' and the date and 
            time. Defaults to None.   
        automatic (bool): whether to automatically iterate through the project
            stages (True) or whether it must be iterating manually (False). 
            Defaults to True.
        library (ClassVar[nodes.Library]): a class attribute containing a 
            dot-accessible dictionary of base classes. Each base class has 
            'subclasses' and 'instances' class attributes which contain catalogs
            of subclasses and instances of those library classes. This 
            attribute is inherited from Keystone. Changing this attribute will 
            entirely replace the existing links between this instance and all 
            other base classes.
            
    """
    name: Optional[str] = None
    settings: Optional[configuration.Settings] = None
    clerk: Optional[filing.Clerk] = None
    director: Optional[helpers.Director] = None
    outline: Optional[Union[Type[stages.Outline], stages.Outline]] = None
    workflow: Optional[Union[Type[stages.Workflow], stages.Workflow]] = None
    product: Optional[Union[Type[stages.Product], stages.Product]] = None
    library: helpers.ProjectLibrary = dataclasses.field(
        default_factory = helpers.ProjectLibrary)
    data: Optional[object] = None
    identification: Optional[str] = None
    automatic: bool = True
    
    """ Initialization Methods """

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Removes various python warnings from console output.
        warnings.filterwarnings('ignore')
        # Calls parent and/or mixin initialization method(s).
        try:
            super().__post_init__()
        except AttributeError:
            pass
        # Calls validation methods.
        for validation in self._validations:
            try:
                getattr(self, f'_validate_{validation}')()
            except AttributeError:
                pass
        # Sets multiprocessing technique, if necessary.
        self._set_parallelization()
        # Calls 'complete' if 'automatic' is True.
        if self.automatic:
            self.complete()

    """ Public Methods """

    @classmethod
    def from_settings(
        cls, 
        settings: configuration.Settings, 
        **kwargs) -> Project:
        """[summary]

        Args:
            settings (SettingsSources): [description]

        Returns:
            Project: [description]
            
        """        
        
        if isinstance(settings, amos.shared.bases.settings):
            outline = amos.shared.bases.outline.create(source = settings)
        elif (inspect.isclass(settings) 
              and issubclass(settings, amos.shared.bases.settings)):
            outline = amos.shared.bases.outline.create(source = settings())
        else:
            settings = amos.shared.bases.settings.create(source = settings)
            outline = amos.shared.bases.outline.create(source = settings)
        return cls(outline = outline, **kwargs)
        
    """ Public Methods """
    
    def advance(self) -> None:
        """Iterates through next stage."""
        return self.__next__()

    def complete(self) -> None:
        """Iterates through all stages."""
        for stage in self.director.stages:
            self.advance()
        return self
                     
    """ Private Methods """
    
    def _store_shared_settings(self) -> None:
        """[summary]

        Returns:
            [type]: [description]
            
        """
        attributes = dir(amos.shared)
        constants = [a.is_upper() for a in attributes]
        relevant = ['general', 'denovo', 'amos', self.name]
        sections = {k: v for k, v in self.settings.items() if k in relevant}
        constant_keys = [k for k in sections.keys() if k.upper() in constants]
        for key in constant_keys:
            setattr(amos.shared, key.upper(), sections[key])
        return self
                  
    def _validate_outline(self) -> None:
        """Validates the 'outline' attribute.
        
        If 'outline' is None, the default 'outline' in 'configuration' is
        used.
        
        """
        if self.outline is None:
            self.outline = stages.Outline()
        elif isinstance(self.outline, (str, pathlib.Path, dict)):
            self.outline = amos.create(source = self.outline)
        elif isinstance(self.outline, amos.shared.bases.settings):
            if not isinstance(self.outline, amos.shared.bases.outline):
                self.outline = amos.shared.bases.outline.create(
                    source = self.outline.contents)
        else:
            raise TypeError('outline must be a Settings, str, pathlib.Path, '
                            'dict, or None type')
        return self      
    
    def _validate_identification(self) -> None:
        """Creates unique 'identification' if one doesn't exist.
        
        By default, 'identification' is set to the 'name' attribute followed by
        an underscore and the date and time.
        
        """
        if self.identification is None:
            self.identification = (
                denovo.tools.datetime_string(prefix = self.name))
        elif not isinstance(self.identification, str):
            raise TypeError('identification must be a str or None type')
        return self
    
    def _validate_clerk(self) -> None:
        """Creates or validates 'clerk'."""
        if isinstance(self.clerk, (str, pathlib.Path, type(None))):
            self.clerk = amos.shared.bases.clerk(settings = self.outline)
        elif isinstance(self.clerk, amos.shared.bases.clerk):
            self.clerk.settings = self.outline
            self.clerk._add_settings()
        else:
            raise TypeError('clerk must be a Clerk, str, pathlib.Path, or None '
                            'type')
        return self

    def _validate_director(self) -> None:
        """Creates or validates 'director'."""
        if self.director is None:
            self.director = amos.shared.bases.director(project = self)
        elif not isinstance(self.director, amos.shared.bases.director):
            raise TypeError('director must be a Director or None type')
        return self
    
    def _validate_workflow(self) -> None:
        """Creates or validates 'library'."""
        if self.workflow is None:
            self.workflow = amos.shared.bases.workflow(project = self)
        elif not isinstance(self.workflow, amos.shared.bases.workflow):
            raise TypeError('workflow must be a Workflow or None type')
        return self

    def _set_parallelization(self) -> None:
        """Sets multiprocessing method based on 'outline'."""
        if amos.shared.PARALLELIZE and not globals()['multiprocessing']:
            import multiprocessing
            multiprocessing.set_start_method('spawn') 
        return self
         
    """ Dunder Methods """
      
    def __iter__(self) -> Iterable:
        """Returns iterable of a Project's Director instance.
        
        Returns:
            Iterable: of a Project's Director instance.
            
        """
        return iter(self.director)
 
    def __next__(self) -> None:
        """Completes a stage in 'director'."""
        try:
            next(self.director)
        except StopIteration:
            pass
        return self

@dataclasses.dataclass
class Director(collections.abc.Iterator):
    """Iterator for amos Project instances.
    
    
    """
    project: amos.Project = None
    workshop: ModuleType = denovo.project.workshop

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
        
    """ Properties """
    
    @property
    def current(self) -> str:
        return list(self.stages.keys())[self.index]
    
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
        for stage in self.project.stages:
            self.advance()
        return self
        
    # def functionify(self, source: str, product: str) -> str:
    #     """[summary]

    #     Args:
    #         source (str): [description]
    #         product (str): [description]

    #     Returns:
    #         str: [description]
            
    #     """        
    #     name = f'{source}_to_{product}'
    #     return getattr(self.workshop, name)

    # def kwargify(self, func: Callable) -> dict[Hashable, Any]:
    #     """[summary]

    #     Args:
    #         func (Callable): [description]

    #     Returns:
    #         dict[Hashable, Any]: [description]
            
    #     """        
    #     parameters = inspect.signature(func).parameters.keys()
    #     kwargs = {}
    #     for parameter in parameters:
    #         try:
    #             kwargs[parameter] = getattr(self.project, parameter)
    #         except AttributeError:
    #             pass
    #     return kwargs
    
    """ Dunder Methods """

    # def __getattr__(self, attribute: str) -> Any:
    #     """[summary]

    #     Args:
    #         attribute (str): [description]

    #     Raises:
    #         IndexError: [description]

    #     Returns:
    #         Any: [description]
            
    #     """
    #     if attribute in self.stages:
    #         if attribute == self.subsequent:
    #             self.__next__()
    #         else:
    #             raise IndexError(
    #                 f'You cannot call {attribute} because the current stage is '
    #                 f'{self.current} and the next callable stage is '
    #                 f'{self.subsequent}')  
    #     else:
    #         raise KeyError(f'{attribute} is not in {self.__class__.__name__}')             
            
    def __iter__(self) -> Iterable:
        """Returns iterable of a Project instance.
        
        Returns:
            Iterable: of the Project instance.
            
        """
        return self
 
    def __next__(self) -> None:
        """Completes a Stage instance."""
        if self.index + 1 < len(self.stages):
            source = self.stages[self.current]
            product = self.stages[self.subsequent]
            # director = self.functionify(source = source, product = product)
            director = getattr(self.workshop, f'create_{product}')
            if hasattr(configuration, 'VERBOSE') and configuration.VERBOSE:
                print(f'Creating {product}')
            kwargs = {'project': self.project}
            setattr(self.project, product, director(**kwargs))
            self.index += 1
            if hasattr(configuration, 'VERBOSE') and configuration.VERBOSE:
                print(f'Completed {product}')
        else:
            raise StopIteration
        return self


@dataclasses.dataclass
class Director(Iterator):
    """Iterator for amos Project instances.
    
    
    """
    project: Project = None
    workshop: ModuleType = denovo.project.workshop

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
        
    """ Properties """
    
    @property
    def current(self) -> str:
        return list(self.stages.keys())[self.index]
    
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
        for stage in self.project.stages:
            self.advance()
        return self
        
    # def functionify(self, source: str, product: str) -> str:
    #     """[summary]

    #     Args:
    #         source (str): [description]
    #         product (str): [description]

    #     Returns:
    #         str: [description]
            
    #     """        
    #     name = f'{source}_to_{product}'
    #     return getattr(self.workshop, name)

    # def kwargify(self, func: Callable) -> dict[Hashable, Any]:
    #     """[summary]

    #     Args:
    #         func (Callable): [description]

    #     Returns:
    #         dict[Hashable, Any]: [description]
            
    #     """        
    #     parameters = inspect.signature(func).parameters.keys()
    #     kwargs = {}
    #     for parameter in parameters:
    #         try:
    #             kwargs[parameter] = getattr(self.project, parameter)
    #         except AttributeError:
    #             pass
    #     return kwargs
    
    """ Dunder Methods """

    # def __getattr__(self, attribute: str) -> Any:
    #     """[summary]

    #     Args:
    #         attribute (str): [description]

    #     Raises:
    #         IndexError: [description]

    #     Returns:
    #         Any: [description]
            
    #     """
    #     if attribute in self.stages:
    #         if attribute == self.subsequent:
    #             self.__next__()
    #         else:
    #             raise IndexError(
    #                 f'You cannot call {attribute} because the current stage is '
    #                 f'{self.current} and the next callable stage is '
    #                 f'{self.subsequent}')  
    #     else:
    #         raise KeyError(f'{attribute} is not in {self.__class__.__name__}')             
            
    def __iter__(self) -> Iterable:
        """Returns iterable of a Project instance.
        
        Returns:
            Iterable: of the Project instance.
            
        """
        return self
 
    def __next__(self) -> None:
        """Completes a Stage instance."""
        if self.index + 1 < len(self.stages):
            source = self.stages[self.current]
            product = self.stages[self.subsequent]
            # director = self.functionify(source = source, product = product)
            director = getattr(self.workshop, f'create_{product}')
            if hasattr(configuration, 'VERBOSE') and configuration.VERBOSE:
                print(f'Creating {product}')
            kwargs = {'project': self.project}
            setattr(self.project, product, director(**kwargs))
            self.index += 1
            if hasattr(configuration, 'VERBOSE') and configuration.VERBOSE:
                print(f'Completed {product}')
        else:
            raise StopIteration
        return self