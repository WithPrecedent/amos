"""
core: classes for easy introspection
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
    Inspector (object):
    ClassInspector (Inspector):
    InstanceInspector (Inspector):
    ModuleInspector (Inspector):
    PackageInspector (Inspector):
  
ToDo:


"""
from __future__ import annotations
import dataclasses
import inspect
import pathlib
import types
from typing import Any, Optional, Type, Union

from . import module
from . import package
from . import traits


@dataclasses.dataclass
class Inspector(object):
    """Inspector factory which returns the appropraite Inspector subclass.
    
    Args:
        item (Any): unknown item to examine.
                        
    """
    item: Any
    
    """ Initialization Methods """
    
    def __new__(cls, item: Any, *args: Any, **kwargs: Any) -> None:
        """Returns Inspector subclass based on type of 'item'."""
        if isinstance(item, types.ModuleType):
            return ModuleInspector(module = item, *args, **kwargs)
        elif isinstance(item, (pathlib.Path, str)):
            return PackageInspector(folder = item, *args, **kwargs)
        elif inspect.isclass(item):
            return ClassInspector(item = item, *args, **kwargs)
        elif isinstance(item, object):
            return InstanceInspector(item = item, *args, **kwargs)
        else:
            raise TypeError(f'item must be a module, path, class, or object')
    
    """ Properties """
        
    @property
    def attributes(self) -> dict[str, Any]:
        """dict of attribute names and values in 'item'.

        Returns:
            dict[str, Any]: keys are attribute names and values are attribute 
                values.
            
        """
        return traits.get_attributes(
            item = self.item, 
            include_private = self.include_private)
        
    @property
    def contains(self) -> Optional[Union[
        tuple[Type[Any], ...], 
        tuple[tuple[Type[Any], ...], tuple[Type[Any], ...]]]]:
        """Types that 'item' contains.
        
        Returns:
            Optional[Union[tuple[Type[Any], ...], tuple[tuple[Type[Any], ...], 
                tuple[Type[Any], ...]]]]:: returns the types of things contained 
                in 'item'. Returns None if 'item' is not a container.
            
        """
        return traits.has_types(item = self.item)
                
    @property
    def name(self) -> Optional[str]:
        """str name of 'item'.
        
        Returns:
            str: inferred name of the stored 'item'.
            
        """
        return traits.get_name(item = self.item) 
        
    @property
    def type(self) -> Type[Any]:
        """Data type of 'item'.

        Returns:
            Type[Any]: type of the stored 'item'.
            
        """
        return type(self.item)
        
    @property
    def variables(self) -> dict[str, Any]:
        """dict of variable names and variable values in 'item'.
        
        'variables' are all attributes that are neither methods nor properties.

        Returns:
            dict[str, Any]: keys are variable names and values are variable 
                values.
            
        """
        return traits.get_variables(
            item = self.item, 
            include_private = self.include_private)                  


@dataclasses.dataclass
class ClassInspector(Inspector):
    """Inspector for accessing class information from 'item'.
    
    Args:
        item (Type[Any]): class to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to True.
                        
    """
    item: Type[Any]
    include_private: bool = True
                       
    """ Properties """
    
    @property
    def annotations(self) -> dict[str, Type[Any]]:
        """dict of parameters and annotated type hints in 'item'.

        Returns:
            dict[str, Type[Any]]: keys are parameter/attribute names and values
                are type hints.
            
        """
        return traits.get_annotations(
            item = self.item, 
            include_private = self.include_private) 
        
    @property
    def attributes(self) -> list[str]:
        """Attribute names in 'item'.

        Returns:
            list[str]: names of attributes.
            
        """
        return traits.name_attributes(
            item = self.item, 
            include_private = self.include_private)

    @property
    def methods(self) -> dict[str, types.MethodType]:
        """dict of method names and methods in 'item'.

        Returns:
            dict[str, types.MethodType]: keys are method names and values are 
                methods.
            
        """
        return traits.get_methods(
            item = self.item, 
            include_private = self.include_private)

    @property
    def parameters(self) -> list[str]:
        """Names of parameters from a dataclass: 'item'.
        
        Returns:
            list[str]: names of parameters for a dataclass.
            
        """
        return traits.name_parameters(item = self.item) 
    
    @property
    def properties(self) -> list[str]:
        """Property names in 'item'.

        Returns:
            list[str]: names of properties.
            
        """
        return traits.get_properties(
            item = self.item, 
            include_private = self.include_private)           
                 
    @property
    def signatures(self) -> dict[str, inspect.Signature]:
        """dict of method names and signatures in 'item'.

        Returns:
            dict[str, inspect.Signature]: keys are method names and values are 
                signatures for those methods.
            
        """
        return traits.get_signatures(
            item = self.item, 
            include_private = self.include_private)  

    @property
    def variables(self) -> list[str]:
        """Variable names in 'item'.
        
        'variables' are all attributes that are neither methods nor properties.

        Returns:
            list[str]: names of variables in 'item'.
            
        """
        return traits.get_variables(
            item = self.item, 
            include_private = self.include_private)      


@dataclasses.dataclass
class InstanceInspector(Inspector):
    """Inspector for accessing instance information from 'item'.
    
    Args:
        item (Type[Any]): instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to True.
                        
    """
    item: object
    include_private: bool = True
             
    """ Properties """
    
    @property
    def annotations(self) -> dict[str, Type[Any]]:
        """dict of parameters and annotated type hints in 'item'.

        Returns:
            dict[str, Type[Any]]: keys are parameter/attribute names and values
                are type hints.
            
        """
        return traits.get_annotations(
            item = self.item, 
            include_private = self.include_private) 

    @property
    def methods(self) -> dict[str, types.MethodType]:
        """dict of method names and methods in 'item'.

        Returns:
            dict[str, types.MethodType]: keys are method names and values are 
                methods.
            
        """
        return traits.get_methods(
            item = self.item, 
            include_private = self.include_private)

    @property
    def parameters(self) -> list[str]:
        """Names of parameters from a dataclass in 'item'.
        
        Returns:
            list[str]: names of parameters for a dataclass.
            
        """
        return traits.name_parameters(item = self.item) 
    
    @property
    def properties(self) -> dict[str, Any]:
        """dict of property names and property values in 'item'.

        Returns:
            dict[str, Any]: keys are property names and values are property 
                values.
            
        """
        return traits.get_properties(
            item = self.item, 
            include_private = self.include_private)           
                 
    @property
    def signatures(self) -> dict[str, inspect.Signature]:
        """dict of method names and signatures in 'item'.

        Returns:
            dict[str, inspect.Signature]: keys are method names and values are 
                signatures for those methods.
            
        """
        return traits.get_signatures(
            item = self.item, 
            include_private = self.include_private)                
         

@dataclasses.dataclass
class ModuleInspector(Inspector):
    """Inspector for accessing module information from 'item'.
    
    Args:
        item (types.ModuleType): module to inspect.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (True). Defauls to False.
                        
    """
    item: types.ModuleType
    include_private: bool = True
             
    """ Properties """
        
    @property
    def classes(self) -> dict[str, Type[Any]]:
        """dict of class names and classes in 'item'.

        Returns:
            dict[str, Type[Any]: keys are class names and values are classes.
            
        """
        return module.get_classes(
            item = self.item, 
            include_private = self.include_private)
        
    @property
    def functions(self) -> dict[str, types.FunctionType]:
        """dict of function names and functions in 'item'.

        Returns:
            dict[str, types.FunctionType]: keys are function names and values 
                are functions.
            
        """
        return module.get_functions(
            item = self.item, 
            include_private = self.include_private)
                 
    @property
    def signatures(self) -> dict[str, inspect.Signature]:
        """dict of method names and method signatures in 'item'.

        Returns:
            dict[str, inspect.Signature]: keys are method names and values are 
                signatures for those methods.
            
        """
        return traits.get_signatures(
            item = self.item, 
            include_private = self.include_private)  
          

@dataclasses.dataclass
class PackageInspector(Inspector):
    """Inspector for accessing package information from 'item'.
    
    Attributes:
        item (Union[pathlib.Path, str]): folder for which information should
            be made available.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to True.
        include_subfolders (bool): whether to include subitems in the package.
            Defaults to True. 
            
    """
    item: Union[pathlib.Path, str]
    include_private: bool = True
    include_subfolders: bool = True

    """ Initialization Methods """
    
    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        self.item = convert.pathlibify(item = self.item)
        
    """ Properties """

    @property
    def files(self) -> list[pathlib.Path]:
        """Non-python-module file paths in 'item'.

        Returns:
            list[pathlib.Path]: list of non-python-module file paths.
            
        """
        return package.get_file_paths(
            item = self.item, 
            recursive = self.include_subfolders)

    @property    
    def folders(self) -> list[pathlib.Path]:
        """Folder paths in 'item'.

        Returns:
            list[pathlib.Path]: list of folder paths.
            
        """
        return package.get_folder_paths(
            item = self.item, 
            recursive = self.include_subfolders)

    @property          
    def modules(self) -> dict[str, types.ModuleType]:
        """dict of python module names and modules in 'item'.
        
        Args:
            item (Union[str, pathlib.Path]): path of folder to examine.
            
        Returns:
            dict[str, types.ModuleType]: dict with str key names of python 
                modules and values as the corresponding modules.
            
        """
        return package.get_modules(
            item = self.item, 
            recursive = self.include_subfolders)  
        
    @property          
    def module_paths(self) -> list[pathlib.Path]:
        """Python module file paths in 'item'.

        Returns:
            list[pathlib.Path]: list of python-module file paths.
            
        """
        return package.get_module_paths(
            item = self.item, 
            recursive = self.include_subfolders)  
              
    @property    
    def paths(self) -> list[pathlib.Path]:
        """All paths in 'item'.

        Returns:
            list[pathlib.Path]: list of all paths.
            
        """
        return package.get_paths(
            item = self.item, 
            recursive = self.include_subfolders)
