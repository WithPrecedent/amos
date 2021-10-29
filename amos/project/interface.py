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
    Hashable, Iterable, Mapping, MutableMapping, MutableSequence, Sequence)
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
    name: str = None
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
    def from_settings(cls, 
                      settings: amos.shared.SettingsSources, 
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
            self.outline = amos.shared.bases.outline()
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
class Parameters(denovo.base.Lexicon):
    """Creates and stores parameters for a Component.
    
    The use of Parameters is entirely optional, but it provides a handy tool
    for aggregating data from an array of sources, including those which only 
    become apparent during execution of an amos project, to create a unified 
    set of implementation parameters.
    
    Parameters can be unpacked with '**', which will turn the 'contents' 
    attribute an ordinary set of kwargs. In this way, it can serve as a drop-in
    replacement for a dict that would ordinarily be used for accumulating 
    keyword arguments.
    
    If an amos class uses a Parameters instance, the 'finalize' method should
    be called before that instance's 'implement' method in order for each of the
    parameter types to be incorporated.
    
    Args:
        contents (Mapping[str, Any]): keyword parameters for use by an amos
            classes' 'implement' method. The 'finalize' method should be called
            for 'contents' to be fully populated from all sources. Defaults to
            an empty dict.
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout amos. To properly match parameters
            in an Outline instance, 'name' should be the prefix to "_parameters"
            as a section name in an Outline instance. Defaults to None. 
        default (Mapping[str, Any]): default parameters that will be used if 
            they are not overridden. Defaults to an empty dict.
        implementation (Mapping[str, str]): parameters with values that can only 
            be determined at runtime due to dynamic nature of amos and its 
            workflows. The keys should be the names of the parameters and the 
            values should be attributes or items in 'contents' of 'project' 
            passed to the 'finalize' method. Defaults to an emtpy dict.
        selected (Sequence[str]): an exclusive list of parameters that are 
            allowed. If 'selected' is empty, all possible parameters are 
            allowed. However, if any are listed, all other parameters that are
            included are removed. This is can be useful when including 
            parameters in an Outline instance for an entire step, only some of
            which might apply to certain techniques. Defaults to an empty list.

    """
    contents: Mapping[str, Any] = dataclasses.field(default_factory = dict)
    name: str = None
    default: Mapping[str, Any] = dataclasses.field(default_factory = dict)
    implementation: Mapping[str, str] = dataclasses.field(
        default_factory = dict)
    selected: Sequence[str] = dataclasses.field(default_factory = list)
      
    """ Public Methods """

    def finalize(self, project: amos.Project, **kwargs) -> None:
        """Combines and selects final parameters into 'contents'.

        Args:
            project (amos.Project): instance from which implementation and 
                settings parameters can be derived.
            
        """
        # Uses kwargs and 'default' parameters as a starting base.
        parameters = self.default
        parameters.update(kwargs)
        # Adds any parameters from 'settings'.
        try:
            parameters.update(self._from_settings(settings = project.settings))
        except AttributeError:
            pass
        # Adds any implementation parameters.
        if self.implementation:
            parameters.update(self._at_runtime(project = project))
        # Adds any parameters already stored in 'contents'.
        parameters.update(self.contents)
        # Limits parameters to those in 'selected'.
        if self.selected:
            self.contents = {k: self.contents[k] for k in self.selected}
        self.contents = parameters
        return self

    """ Private Methods """
     
    def _from_settings(self, 
        settings: amos.options.Outline) -> dict[str, Any]: 
        """Returns any applicable parameters from 'settings'.

        Args:
            settings (amos.options.Outline): instance with possible 
                parameters.

        Returns:
            dict[str, Any]: any applicable settings parameters or an empty dict.
            
        """
        try:
            parameters = settings[f'{self.name}_parameters']
        except KeyError:
            suffix = self.name.split('_')[-1]
            prefix = self.name[:-len(suffix) - 1]
            try:
                parameters = settings[f'{prefix}_parameters']
            except KeyError:
                try:
                    parameters = settings[f'{suffix}_parameters']
                except KeyError:
                    parameters = {}
        return parameters
   
    def _at_runtime(self, project: amos.Project) -> dict[str, Any]:
        """Adds implementation parameters to 'contents'.

        Args:
            project (amos.Project): instance from which implementation 
                parameters can be derived.

        Returns:
            dict[str, Any]: any applicable settings parameters or an empty dict.
                   
        """    
        for parameter, attribute in self.implementation.items():
            try:
                self.contents[parameter] = getattr(project, attribute)
            except AttributeError:
                try:
                    self.contents[parameter] = project.contents[attribute]
                except (KeyError, AttributeError):
                    pass
        return self




@dataclasses.dataclass
class Stage(denovo.framework.Keystone, abc.ABC):
    
    pass
    
    

@dataclasses.dataclass
class Worker(denovo.quirks.Element, collections.abc.Iterable, abc.ABC):
    """Keystone class for parts of an amos workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout amos. For example, if an amos 
            instance needs settings from an Outline instance, 'name' should 
            match the appropriate section name in an Outline instance. Defaults 
            to None. 
        contents (dict[str, list[str]]): an adjacency list where the keys are 
            names of components and the values are names of components which the 
            key component is connected to. Defaults to an empty dict.
        parameters (MutableMapping[Hashable, Any]): parameters to be attached to 
            'contents' when the 'implement' method is called. Defaults to an 
            empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        default (Any): default value to return when the 'get' method is used.
            Defaults to an empty list.

    Attributes:
        library (ClassVar[Library]): library that stores concrete (non-abstract) 
            subclasses and instances of Component. 
                              
    """
    name: str = None
    contents: dict[str, list[str]] = dataclasses.field(default_factory = dict)
    parameters: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = lambda: Parameters)
    iterations: Union[int, str] = 1

    """ Public Methods """  
    
    def implement(self, project: amos.Project, **kwargs) -> amos.Project:
        """Applies 'contents' to 'project'.
        
        Args:
            project (amos.Project): instance from which data needed for 
                implementation should be derived and all results be added.

        Returns:
            amos.Project: with possible changes made.
            
        """
        return self._implement_in_serial(project = project, **kwargs)    

    def organize(self, subcomponents: dict[str, list[str]]) -> None:
        """[summary]

        Args:
            subcomponents (dict[str, list[str]]): [description]

        """
        subcomponents = self._serial_order(
            name = self.name, 
            subcomponents = subcomponents)
        nodes = list(more_itertools.collapse(subcomponents))
        if nodes:
            self.extend(nodes = nodes)
        return self       

    """ Private Methods """

    def _implement_in_serial(self, 
                             project: amos.Project,
                             **kwargs) -> amos.Project:
        """Applies stored nodes to 'project' in order.

        Args:
            project (Project): amos project to apply changes to and/or
                gather needed data from.
                
        Returns:
            Project: with possible alterations made.       
        
        """
        for node in self.paths[0]:
            project = node.execute(project = project, **kwargs)
        return project
       
    def _serial_order(self, 
                      name: str,
                      subcomponents: dict[str, list[str]]) -> list[Hashable]:
        """[summary]

        Args:
            name (str): [description]
            directive (Directive): [description]

        Returns:
            list[Hashable]: [description]
            
        """   
        organized = []
        components = subcomponents[name]
        for item in components:
            organized.append(item)
            if item in subcomponents:
                organized_subcomponents = []
                subcomponents = self._serial_order(
                    name = item, 
                    subcomponents = subcomponents)
                organized_subcomponents.append(subcomponents)
                if len(organized_subcomponents) == 1:
                    organized.append(organized_subcomponents[0])
                else:
                    organized.append(organized_subcomponents)
        return organized   
  