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
from collections.abc import Iterable, Sequence
import dataclasses
from typing import Any, ClassVar, Optional, Type, Union
import warnings

from ..core import composites
from ..report import clock
from . import options


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
    settings: options.SETTINGS = dataclasses.field(
        default_factory = options.SETTINGS)
    clerk: options.CLERK = dataclasses.field(
        default_factory = options.CLERK)
    director: options.DIRECTOR = dataclasses.field(
        default_factory = options.DIRECTOR)
    stages: Sequence[Union[str, Type[options.STAGE], options.STAGE]] = (
        dataclasses.field(default_factory = list))
    library: options.LIBRARY = dataclasses.field(
        default_factory = options.LIBRARY)
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
        self._validate_settings()
        self._validate_identification
        self._validate_clerk()
        self._validate_director()
        self._validate_data()
        # Sets multiprocessing technique, if necessary.
        self._set_parallelization()
        # Calls 'complete' if 'automatic' is True.
        if self.automatic:
            self.complete()
        
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

    def _validate_settings(self) -> None:
        """Creates or validates 'settings'."""
        options.SETTINGS.create(item = self.settings)
    
    def _validate_identification(self) -> None:
        """Creates unique 'identification' if one doesn't exist.
        
        By default, 'identification' is set to the 'name' attribute followed by
        an underscore and the date and time.
        
        """
        if self.identification is None:
            prefix = self.name + '_'
            self.identification = clock.datetime_string(prefix = prefix)
        elif not isinstance(self.identification, str):
            raise TypeError('identification must be a str or None type')
        return self
          
    def _validate_clerk(self) -> None:
        """Creates or validates 'clerk'."""
        options.CLERK.create(item = self.clerk, settings = self.settings)
        return self

    def _validate_director(self) -> None:
        """Creates or validates 'director'."""
        options.DIRECTOR.create(project = self)
        return self
    
    def _validate_director(self) -> None:
        """Creates or validates 'data'."""
        return self

    def _set_parallelization(self) -> None:
        """Sets multiprocessing method based on 'settings'."""
        if self.settings['general']['parallelize']:
            if not globals()['multiprocessing']:
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
