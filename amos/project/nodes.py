"""
nodes: project workflow nodes and related classes
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
    Callable, Hashable, Mapping, MutableMapping, MutableSequence, Sequence)
import copy
import dataclasses
import inspect
import multiprocessing
from typing import Any, ClassVar, Optional, Type, Union

from ..base import mappings
from ..repair import convert

@dataclasses.dataclass
class Component(abc.ABC):
    """Base class for nodes in a project workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout amicus. For example, if an amicus 
            instance needs settings from an Outline instance, 'name' should 
            match the appropriate section name in an Outline instance. Defaults 
            to None. 
        contents (Any): stored item(s) to be used by the 'implement' method.
            Defaults to None.
        parameters (MutableMapping[Hashable, Any]): parameters to be attached to 
            'contents' when the 'implement' method is called. Defaults to an 
            empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.

    Attributes:
        library (ClassVar[Library]): library that stores concrete (non-abstract) 
            subclasses and instances of Component. 
  
    """
    name: str = None
    contents: Any = None
    parameters: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = Parameters)
    iterations: Union[int, str] = 1
    library: ClassVar[Library] = Library()

    """ Initialization Methods """
    
    def __init_subclass__(cls, **kwargs):
        """Adds 'cls' to 'library'."""
        super().__init_subclass__(**kwargs)
        # Adds concrete subclasses to 'library'.
        if not abc.ABC in cls.__bases__:
            cls.library.register(component = cls)
            
    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Calls parent and/or mixin initialization method(s).
        try:
            super().__post_init__()
        except AttributeError:
            pass
        # Adds instance to 'library'.
        self.library.register(component = self)
  
    """ Required Subclass Methods """

    @abc.abstractmethod
    def implement(self, project: amicus.Project, **kwargs) -> amicus.Project:
        """Applies 'contents' to 'project'.

        Subclasses must provide their own methods.

        Args:
            project (amicus.Project): instance from which data needed for 
                implementation should be derived and all results be added.

        Returns:
            amicus.Project: with possible changes made.
            
        """
        pass

    """ Public Methods """
    
    def execute(self, 
        project: amicus.Project, 
        iterations: Union[int, str] = None, 
        **kwargs) -> amicus.Project:
        """Calls the 'implement' method the number of times in 'iterations'.

        Args:
            project (amicus.Project): instance from which data needed for 
                implementation should be derived and all results be added.

        Returns:
            amicus.Project: with possible changes made.
            
        """
        if iterations is None:
            iterations = self.iterations
        if self.contents not in [None, 'None', 'none']:
            if self.parameters:
                if isinstance(self.parameters, Parameters):
                    self.parameters.finalize(project = project)
                parameters = self.parameters
                parameters.update(kwargs)
            else:
                parameters = kwargs
            if iterations in ['infinite']:
                while True:
                    project = self.implement(project = project, **parameters)
            else:
                for _ in range(iterations):
                    project = self.implement(project = project, **parameters)
        return project

    """ Dunder Methods """
    
    def __call__(self, project: amicus.Project, **kwargs) -> amicus.Project:
        """Applies 'execute' method if an instance is called.
        
        Args:
            project (amicus.Project): instance from which data needed for 
                implementation should be derived and all results be added.

        Returns:
            amicus.Project: with possible changes made.
            
        """
        return self.execute(project = project, **kwargs)


@dataclasses.dataclass
class Task(denovo.structures.Node):
    """Node type for fiat Workflows.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout fiat. For example, if a fiat 
            instance needs settings from an Outline instance, 'name' should 
            match the appropriate section name in an Outline instance. 
            Defaults to None. 
        step (Union[str, Callable]):
        

    """
    name: str = None
    step: Union[str, Callable] = None
    technique: Callable = None
      
    """ Required Subclass Methods """
    
    @abc.abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass
  
    """ Public Methods """
    
    def implement(self, project: amicus.Project, **kwargs) -> amicus.Project:
        """Applies 'contents' to 'project'.

        Args:
            project (amicus.Project): instance from which data needed for 
                implementation should be derived and all results be added.

        Returns:
            amicus.Project: with possible changes made.
            
        """
        project = self.contents.execute(project = project, **kwargs)
        return project    


@dataclasses.dataclass
class Step(amicus.base.Proxy, Task):
    """Wrapper for a Technique.

    Subclasses of Step can store additional methods and attributes to implement
    all possible technique instances that could be used. This is often useful 
    when creating branching, parallel workflows which test a variety of 
    strategies with similar or identical parameters and/or methods.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout amicus. For example, if an amicus 
            instance needs settings from an Outline instance, 'name' should 
            match the appropriate section name in an Outline instance. Defaults 
            to None. 
        contents (Technique): stored Technique instance to be used by the 
            'implement' method. Defaults to None.
        parameters (MutableMapping[Hashable, Any]): parameters to be attached to 
            'contents' when the 'implement' method is called. Defaults to an 
            empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.

    Attributes:
        library (ClassVar[Library]): library that stores concrete (non-abstract) 
            subclasses and instances of Component. 
                                                 
    """
    name: str = None
    contents: Technique = None
    parameters: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = Parameters)
    iterations: Union[int, str] = 1
                    
    """ Properties """
    
    @property
    def technique(self) -> Technique:
        return self.contents
    
    @technique.setter
    def technique(self, value: Technique) -> None:
        self.contents = value
        return self
    
    @technique.deleter
    def technique(self) -> None:
        self.contents = None
        return self
    
    """ Public Methods """
    
    def organize(self, technique: Technique) -> Technique:
        """[summary]

        Args:
            technique (Technique): [description]

        Returns:
            Technique: [description]
            
        """
        if self.parameters:
            new_parameters = self.parameters
            new_parameters.update(technique.parameters)
            technique.parameters = new_parameters
        return technique
        
                                                  
@dataclasses.dataclass
class Technique(Task):
    """Keystone class for primitive objects in an amicus composite object.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout amicus. For example, if an amicus 
            instance needs settings from an Outline instance, 'name' should 
            match the appropriate section name in an Outline instance. Defaults 
            to None. 
        contents (Callable): stored Callable algorithm to be used by the 
            'implement' method. Defaults to None.
        parameters (MutableMapping[Hashable, Any]): parameters to be attached to 
            'contents' when the 'implement' method is called. Defaults to an 
            empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.

    Attributes:
        library (ClassVar[Library]): library that stores concrete (non-abstract) 
            subclasses and instances of Component. 
                                                 
    """
    name: str = None
    contents: Callable = None
    parameters: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = Parameters)
    iterations: Union[int, str] = 1
    step: str = None
        
    """ Properties """
    
    @property
    def algorithm(self) -> Union[object, str]:
        return self.contents
    
    @algorithm.setter
    def algorithm(self, value: Union[object, str]) -> None:
        self.contents = value
        return self
    
    @algorithm.deleter
    def algorithm(self) -> None:
        self.contents = None
        return self

    """ Public Methods """

    def execute(self, 
        project: amicus.Project, 
        iterations: Union[int, str] = None, 
        **kwargs) -> amicus.Project:
        """Calls the 'implement' method the number of times in 'iterations'.

        Args:
            project (amicus.Project): instance from which data needed for 
                implementation should be derived and all results be added.

        Returns:
            amicus.Project: with possible changes made.
            
        """
        if self.step is not None:
            step = self.library.instance(name = self.step)
            self = step.organize(technique = self)
        return super().execute(
            project = project, 
            iterations = iterations, 
            **kwargs)

