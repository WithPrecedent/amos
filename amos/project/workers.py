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
import collections.abc
import copy
import dataclasses
import inspect
import multiprocessing
from typing import (Any, Callable, ClassVar, dict, Hashable, Iterable, list, 
                    Mapping, MutableMapping, MutableSequence, Optional, 
                    Sequence, Set, tuple, Type, Union)

import denovo
import more_itertools



@dataclasses.dataclass
class ProjectLibrary(mappings.Library):
    """Stores project classes and class instances.
    
    When searching for matches, instances are prioritized over classes.
    
    Args:
        classes (Catalog): a catalog of stored classes. Defaults to any empty
            Catalog.
        instances (Catalog): a catalog of stored class instances. Defaults to an
            empty Catalog.
                 
    """
    classes: mappings.Catalog = dataclasses.field(
        default_factory = mappings.Catalog)
    instances: mappings.Catalog = dataclasses.field(
        default_factory = mappings.Catalog)

    """ Properties """
    
    @property
    def suffixes(self) -> tuple[str]:
        """Returns all stored names and naive plurals of those names.
        
        Returns:
            tuple[str]: all names with an 's' added in order to create simple 
                plurals combined with the stored keys.
                
        """
        plurals = [key + 's' for key in self.instances.keys()]
        return tuple(list(self.classes.keys()) + plurals)
    
    @property
    def laborers(self) -> tuple[str]:
        kind = configuration.bases.laborer
        instances = [
            k for k, v in self.instances.items() if isinstance(v, kind)]
        subclasses = [
            k for k, v in self.subclasses.items() if issubclass(v, kind)]
        return tuple(instances + subclasses)   
        
    @property
    def manager(self) -> tuple[str]:
        kind = configuration.bases.manager
        instances = [
            k for k, v in self.instances.items() if isinstance(v, kind)]
        subclasses = [
            k for k, v in self.subclasses.items() if issubclass(v, kind)]
        return tuple(instances + subclasses)   
     
    @property
    def tasks(self) -> tuple[str]:
        kind = configuration.bases.task
        instances = [
            k for k, v in self.instances.items() if isinstance(v, kind)]
        subclasses = [
            k for k, v in self.subclasses.items() if issubclass(v, kind)]
        return tuple(instances + subclasses)

    @property
    def workers(self) -> tuple[str]:
        kind = configuration.bases.worker
        instances = [
            k for k, v in self.instances.items() if isinstance(v, kind)]
        subclasses = [
            k for k, v in self.subclasses.items() if issubclass(v, kind)]
        return tuple(instances + subclasses)

    """ Public Methods """
    
    def classify(self, component: str) -> str:
        """[summary]

        Args:
            component (str): [description]

        Returns:
            str: [description]
            
        """        
        if component in self.laborers:
            return 'laborer'
        elif component in self.managers:
            return 'manager'
        elif component in self.tasks:
            return 'task'
        elif component in self.workers:
            return 'worker'
        else:
            raise TypeError(f'{component} is not a recognized type')

    def instance(self, name: Union[str, Sequence[str]], **kwargs) -> Component:
        """Returns instance of first match of 'name' in stored catalogs.
        
        The method prioritizes the 'instances' catalog over 'subclasses' and any
        passed names in the order they are listed.
        
        Args:
            name (Union[str, Sequence[str]]): [description]
            
        Raises:
            KeyError: [description]
            
        Returns:
            Component: [description]
            
        """
        names = convert.iterify(name)
        primary = names[0]
        item = None
        for key in names:
            for catalog in ['instances', 'subclasses']:
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
            instance = item(name = primary, **kwargs)
        else:
            instance = copy.deepcopy(item)
            for key, value in kwargs.items():
                setattr(instance, key, value)  
        return instance 

    def parameterify(self, name: Union[str, Sequence[str]]) -> list[str]:
        """[summary]

        Args:
            name (Union[str, Sequence[str]]): [description]

        Returns:
            list[str]: [description]
            
        """        
        component = self.select(name = name)
        return list(component.__annotations__.keys())
       
    def register(self, component: Union[Component, Type[Component]]) -> None:
        """[summary]

        Args:
            component (Union[Component, Type[Component]]): [description]

        Raises:
            TypeError: [description]

        Returns:
            [type]: [description]
            
        """
        if isinstance(component, Component):
            instances_key = self._get_instances_key(component = component)
            self.instances[instances_key] = component
            subclasses_key = self._get_subclasses_key(component = component)
            if subclasses_key not in self.subclasses:
                self.subclasses[subclasses_key] = component.__class__
        elif inspect.isclass(component) and issubclass(component, Component):
            subclasses_key = self._get_subclasses_key(component = component)
            self.subclasses[subclasses_key] = component
        else:
            raise TypeError(
                f'component must be a Component subclass or instance')
        return self
    
    def select(self, name: Union[str, Sequence[str]]) -> Component:
        """Returns subclass of first match of 'name' in stored catalogs.
        
        The method prioritizes the 'subclasses' catalog over 'instances' and any
        passed names in the order they are listed.
        
        Args:
            name (Union[str, Sequence[str]]): [description]
            
        Raises:
            KeyError: [description]
            
        Returns:
            Component: [description]
            
        """
        names = convert.iterify(name)
        item = None
        for key in names:
            for catalog in ['subclasses', 'instances']:
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
            component = item
        else:
            component = item.__class__  
        return component 
    
    """ Private Methods """
    
    def _get_instances_key(self, 
        component: Union[Component, Type[Component]]) -> str:
        """Returns a snakecase key of the class name.
        
        Returns:
            str: the snakecase name of the class.
            
        """
        try:
            key = component.name 
        except AttributeError:
            try:
                key = modify.snakify(component.__name__) 
            except AttributeError:
                key = modify.snakify(component.__class__.__name__)
        return key
    
    def _get_subclasses_key(self, 
        component: Union[Component, Type[Component]]) -> str:
        """Returns a snakecase key of the class name.
        
        Returns:
            str: the snakecase name of the class.
            
        """
        try:
            key = modify.snakify(component.__name__) 
        except AttributeError:
            key = modify.snakify(component.__class__.__name__)
        return key      


@dataclasses.dataclass
class Laborer(graph.Graph, amos.base.Worker):
    """Keystone class for parts of an denovo workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout denovo. For example, if an denovo 
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
        default_factory = Parameters)
    iterations: Union[int, str] = 1
    default: Any = dataclasses.field(default_factory = list)
   
    """ Public Methods """  

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

    def implement(self, project: denovo.Project, **kwargs) -> denovo.Project:
        """Applies 'contents' to 'project'.
        
        Args:
            project (denovo.Project): instance from which data needed for 
                implementation should be derived and all results be added.

        Returns:
            denovo.Project: with possible changes made.
            
        """
        return self._implement_in_serial(project = project, **kwargs)    

    """ Private Methods """
    
    def _implement_in_serial(self, 
        project: denovo.Project, 
        **kwargs) -> denovo.Project:
        """Applies stored nodes to 'project' in order.

        Args:
            project (Project): denovo project to apply changes to and/or
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

 
@dataclasses.dataclass
class Manager(Worker, abc.ABC):
    """Base class for branching and parallel Workers.
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout denovo. For example, if an denovo 
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
        criteria (Union[Callable, str]): algorithm to use to resolve the 
            parallel branches of the workflow or the name of a Component in 
            'library' to use. Defaults to None.

    Attributes:
        library (ClassVar[Library]): library that stores concrete (non-abstract) 
            subclasses and instances of Component. 
                          
    """
    name: str = None
    contents: dict[str, list[str]] = dataclasses.field(default_factory = dict)
    parameters: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = Parameters)
    iterations: Union[int, str] = 1
    default: Any = dataclasses.field(default_factory = list)
    critera: Union[Callable, str] = None
              
    """ Public Methods """

    def organize(self, subcomponents: dict[str, list[str]]) -> None:
        """[summary]

        Args:
            subcomponents (dict[str, list[str]]): [description]

        """
        step_names = subcomponents[self.name]
        nodes = [subcomponents[step] for step in step_names]
        self.branchify(nodes = nodes)
        return self  
       
    def implement(self, project: denovo.Project, **kwargs) -> denovo.Project:
        """Applies 'contents' to 'project'.
        
        Args:
            project (denovo.Project): instance from which data needed for 
                implementation should be derived and all results be added.

        Returns:
            denovo.Project: with possible changes made.
            
        """
        if len(self.contents) > 1 and project.parallelize:
            project = self._implement_in_parallel(project = project, **kwargs)
        else:
            project = self._implement_in_serial(project = project, **kwargs)
        return project      

    """ Private Methods """
   
    def _implement_in_parallel(self, 
        project: denovo.Project, 
        **kwargs) -> denovo.Project:
        """Applies 'implementation' to 'project' using multiple cores.

        Args:
            project (Project): denovo project to apply changes to and/or
                gather needed data from.
                
        Returns:
            Project: with possible alterations made.       
        
        """
        if project.parallelize:
            with multiprocessing.Pool() as pool:
                project = pool.starmap(
                    self._implement_in_serial, 
                    project, 
                    **kwargs)
        return project 


@dataclasses.dataclass
class Contest(Manager):
    """Resolves a parallel workflow by selecting the best option.

    It resolves a parallel workflow based upon criteria in 'contents'
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout denovo. For example, if an denovo 
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
        criteria (Union[Callable, str]): algorithm to use to resolve the 
            parallel branches of the workflow or the name of a Component in 
            'library' to use. Defaults to None.
            
    Attributes:
        library (ClassVar[Library]): library that stores concrete (non-abstract) 
            subclasses and instances of Component. 
                          
    """
    name: str = None
    contents: dict[str, list[str]] = dataclasses.field(default_factory = dict)
    parameters: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = Parameters)
    iterations: Union[int, str] = 1
    default: Any = dataclasses.field(default_factory = list)
    critera: Callable = None
 
    
@dataclasses.dataclass
class Study(Manager):
    """Allows parallel workflow to continue

    A Study might be wholly passive or implement some reporting or alterations
    to all parallel workflows.
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout denovo. For example, if an denovo 
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
        criteria (Union[Callable, str]): algorithm to use to resolve the 
            parallel branches of the workflow or the name of a Component in 
            'library' to use. Defaults to None.
            
    Attributes:
        library (ClassVar[Library]): library that stores concrete (non-abstract) 
            subclasses and instances of Component. 
                        
    """
    name: str = None
    contents: dict[str, list[str]] = dataclasses.field(default_factory = dict)
    parameters: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = Parameters)
    iterations: Union[int, str] = 1
    default: Any = dataclasses.field(default_factory = list)
    critera: Callable = None
  
    
@dataclasses.dataclass
class Survey(Manager):
    """Resolves a parallel workflow by averaging.

    It resolves a parallel workflow based upon the averaging criteria in 
    'contents'
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout denovo. For example, if an denovo 
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
        criteria (Union[Callable, str]): algorithm to use to resolve the 
            parallel branches of the workflow or the name of a Component in 
            'library' to use. Defaults to None.
            
    Attributes:
        library (ClassVar[Library]): library that stores concrete (non-abstract) 
            subclasses and instances of Component. 
                            
    """
    name: str = None
    contents: dict[str, list[str]] = dataclasses.field(default_factory = dict)
    parameters: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = Parameters)
    iterations: Union[int, str] = 1
    default: Any = dataclasses.field(default_factory = list)
    critera: Callable = None
