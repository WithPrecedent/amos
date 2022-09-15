"""
hybrid: lightweight linear composite data structures
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
    
          
To Do:
    Add methods needed for Composite subclassing.
    
"""
from __future__ import annotations
import abc
from collections.abc import (
    Collection, Hashable, Mapping, MutableMapping, MutableSequence, Sequence)
import dataclasses
from typing import Any, Callable, ClassVar, Optional, Type, TypeVar, Union

from ..containers import mappings
from ..containers import sequences
from ..observe import report
from ..change import modify
from . import core
from ..observe import check
 
 
@dataclasses.dataclass # type: ignore
class Pipeline(sequences.Hybrid, Composite):
    """Base class for pipeline data structures.
    
    Args:
        contents (MutableSequence[proxies.Node]): list of stored proxies.Node instances. 
            Defaults to an empty list.
          
    """
    contents: MutableSequence[proxies.Node] = dataclasses.field(
        default_factory = list)

    """ Properties """

    def endpoint(self) -> proxies.Node:
        """Returns the endpoint of the stored composite object."""
        return self.contents[-1]

    def root(self) -> proxies.Node:
        """Returns the root of the stored composite object."""
        return self.contents[0]
    
    """ Public Methods """
        
    def delete(self, item: Any) -> None:
        """Deletes proxies.Node from the stored composite object.
        
        Args:
            item (Any): proxies.Node or key to the a proxies.Node to delete.
        
        Raises:
            KeyError: if 'item' is not in 'contents'.
            
        """
        self.__delitem__(item)
  
    def merge(item: Composite, *args: Any, **kwargs: Any) -> None:
        """Combines 'item' with the stored composite object.

        Args:
            item (Composite): another Composite object to add to the stored 
                composite object.
                
        """
        pass
   
    def walk(
        self, 
        start: Optional[proxies.Node] = None,
        stop: Optional[proxies.Node] = None, 
        path: Optional[Pipeline] = None,
        return_pipelines: bool = False, 
        *args: Any, 
        **kwargs: Any) -> Pipeline:
        """Returns path in the stored composite object from 'start' to 'stop'.
        
        Args:
            start (Optional[proxies.Node]): proxies.Node to start paths from. Defaults to None.
                If it is None, 'start' should be assigned to one of the roots
                of the Composite.
            stop (Optional[proxies.Node]): proxies.Node to stop paths. Defaults to None. If it 
                is None, 'start' should be assigned to one of the roots of the 
                Composite.
            path (Optional[hybrid.Pipeline]): a path from 'start' to 'stop'. 
                Defaults to None. This parameter is used by recursive methods 
                for determining a path.
            return_pipelines (bool): whether to return a Pipelines instance 
                (True) or a hybrid.Pipeline instance (False). Defaults to True.

        Returns:
            Union[hybrid.Pipeline, hybrid.Pipelines]: path(s) through the 
                Composite object. If multiple paths are possible and 
                'return_pipelines' is False, this method should return a 
                Pipeline that includes all such paths appended to each other. If 
                multiple paths are possible and 'return_pipelines' is True, a 
                Pipelines instance with all of the paths should be returned. 
                Defaults to True.
                            
        """
        return self.contents
    
    """ Dunder Methods """
        
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_pipeline(item = instance)
     
 
@dataclasses.dataclass # type: ignore
class Pipelines(sequences.Hybrid, Composite):
    """Base class a collection of Pipeline instances.
        
    Args:
        contents (MutableSequence[proxies.Node]): list of stored Pipeline instances. 
            Defaults to an empty list.

    """
    contents: MutableSequence[Pipeline] = dataclasses.field(
        default_factory = list)

    """ Properties """

    def endpoint(self) -> Pipeline:
        """Returns the endpoint of the stored composite object."""
        return self.contents[list(self.contents.keys())[-1]]

    def root(self) -> Pipeline:
        """Returns the root of the stored composite object."""
        self.contents[list(self.contents.keys())[0]]
    
    """ Public Methods """
  
    def merge(item: Composite, *args: Any, **kwargs: Any) -> None:
        """Combines 'item' with the stored composite object.

        Args:
            item (Composite): another Composite object to add to the stored 
                composite object.
                
        """
        pass

    def walk(
        self, 
        start: Optional[proxies.Node] = None,
        stop: Optional[proxies.Node] = None, 
        path: Optional[Pipeline] = None,
        return_pipelines: bool = True, 
        *args: Any, 
        **kwargs: Any) -> Union[Pipeline, Pipelines]:
        """Returns path in the stored composite object from 'start' to 'stop'.
        
        Args:
            start (Optional[proxies.Node]): proxies.Node to start paths from. Defaults to None.
                If it is None, 'start' should be assigned to one of the roots
                of the Composite.
            stop (Optional[proxies.Node]): proxies.Node to stop paths. Defaults to None. If it 
                is None, 'start' should be assigned to one of the roots of the 
                Composite.
            path (Optional[hybrid.Pipeline]): a path from 'start' to 'stop'. 
                Defaults to None. This parameter is used by recursive methods 
                for determining a path.
            return_pipelines (bool): whether to return a Pipelines instance 
                (True) or a hybrid.Pipeline instance (False). Defaults to True.

        Returns:
            Union[hybrid.Pipeline, hybrid.Pipelines]: path(s) through the 
                Composite object. If multiple paths are possible and 
                'return_pipelines' is False, this method should return a 
                Pipeline that includes all such paths appended to each other. If 
                multiple paths are possible and 'return_pipelines' is True, a 
                Pipelines instance with all of the paths should be returned. 
                Defaults to True.
                            
        """
        return self.items()
        
    """ Dunder Methods """
        
    @classmethod
    def __instancecheck__(cls, instance: object) -> bool:
        return check.is_pipelines(item = instance)
         