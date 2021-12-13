"""
formats: base file formats
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
    FileFormat (object): contains data needed for a Clerk-compatible file 
        format.
    formats (dict): a dictionary of the out-of-the-box supported file formats.
     
"""
from __future__ import annotations
from collections.abc import Mapping, MutableMapping
import dataclasses
import inspect
import pathlib
import types
from typing import Any, Optional, Type, Union

from ..observe import traits
from . import lazy
 
      
@dataclasses.dataclass
class FileFormat(object):
    """File format information.

    Args:
        name (Optional[str]): the format name which should match the key when a 
            FileFormat instance is stored.
        module (Optional[str]): name of module where object to incorporate is, 
            which can either be a amos or non-amos module. Defaults to 
            None.
        extension (Optional[str]): str file extension to use. Defaults to None.
        load_method (Optional[Union[str, types.FunctionType]]): if a str, it is
            the name of import method in 'module' to use. Otherwise, it should
            be a function for loading. Defaults to None.
        save_method (Optional[Union[str, types.FunctionType]]): if a str, it is
            the name of import method in 'module' to use. Otherwise, it should
            be a function for saved. Defaults to None.
        parameters (Mapping[str, str]]): shared parameters to use from 
            configuration settings where the key is the parameter name that the 
            load or save method should use and the value is the key for the 
            argument in the shared parameters. Defaults to an empty dict. 

    """
    name: Optional[str] = None
    module: Optional[str] = None
    extension: Optional[str] = None
    load_method: Optional[Union[str, types.FunctionType]] = None
    save_method: Optional[Union[str, types.FunctionType]] = None
    parameters: Mapping[str, str] = dataclasses.field(default_factory = dict)
    
    """ Properties """
    
    @property
    def loader(self) -> types.FunctionType:
        """[summary]

        Returns:
            types.FunctionType: [description]
            
        """        
        return self._validate_io_method(method = 'load')
    
    def saver(self) -> types.FunctionType:
        """[summary]

        Returns:
            types.FunctionType: [description]
            
        """        
        return self._validate_io_method(method = 'save')
    
    """ Private Methods """
    
    def _validate_io_method(self, method: str) -> None:
        """[summary]

        Args:
            method (str): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
            
        """        
        attribute = f'{method}_method'
        value = getattr(self, attribute)
        if isinstance(value, str):
            imported = lazy.from_import_path(
                path = value, 
                package = self.module)
            setattr(self, attribute, imported)
            return imported
        if not isinstance(value, types.FunctionType):
            raise ValueError(
                f'{method}_method must be a str, function, or method')
        return self
            
    """ Dunder Methods """
    
    @classmethod
    def __subclasshook__(cls, subclass: Type[Any]) -> bool:
        """Returns whether 'subclass' is a virtual or real subclass.

        Args:
            subclass (Type[Any]): item to test as a subclass.

        Returns:
            bool: whether 'subclass' is a real or virtual subclass.
            
        """
        return (subclass in cls.__subclasses__() 
                or traits.has_attributes(
                    item = subclass,
                    methods = [
                        'name', 'module', 'extension', 'load_method',
                        'save_method', 'parameters']))


""" Included File Formats """

file_formats: MutableMapping[str, FileFormat] = {
    'csv': FileFormat(
        name = 'csv',
        module =  'pandas',
        extension = '.csv',
        load_method = 'read_csv',
        save_method = 'to_csv',
        parameters = {
            'encoding': 'file_encoding',
            'index_col': 'index_column',
            'header': 'include_header',
            'low_memory': 'conserve_memory',
            'nrows': 'test_size'}),
    'excel': FileFormat(
        name = 'excel',
        module =  'pandas',
        extension = '.xlsx',
        load_method = 'read_excel',
        save_method = 'to_excel',
        parameters = {
            'index_col': 'index_column',
            'header': 'include_header',
            'nrows': 'test_size'}),
    'feather': FileFormat(
        name = 'feather',
        module =  'pandas',
        extension = '.feather',
        load_method = 'read_feather',
        save_method = 'to_feather',
        parameters = {'nthreads': 'threads'}),
    'hdf': FileFormat(
        name = 'hdf',
        module =  'pandas',
        extension = '.hdf',
        load_method = 'read_hdf',
        save_method = 'to_hdf',
        parameters = {
            'columns': 'included_columns',
            'chunksize': 'test_size'}),
    'json': FileFormat(
        name = 'json',
        module =  'pandas',
        extension = '.json',
        load_method = 'read_json',
        save_method = 'to_json',
        parameters = {
            'encoding': 'file_encoding',
            'columns': 'included_columns',
            'chunksize': 'test_size'}),
    'stata': FileFormat(
        name = 'stata',
        module =  'pandas',
        extension = '.dta',
        load_method = 'read_stata',
        save_method = 'to_stata',
        parameters = {'chunksize': 'test_size'}),
    'text': FileFormat(
        name = 'text',
        module =  None,
        extension = '.txt',
        load_method = '_import_text',
        save_method = '_export_text'),
    'png': FileFormat(
        name = 'png',
        module =  'seaborn',
        extension = '.png',
        save_method = 'save_fig',
        parameters = {
            'bbox_inches': 'visual_tightness', 
            'format': 'visual_format'}),
    'pickle': FileFormat(
        name = 'pickle',
        module =  None,
        extension = '.pickle',
        load_method = '_pickle_object',
        save_method = '_unpickle_object')}   

def validate_file_format(file_format: Union[str, FileFormat]) -> FileFormat:
    """Selects 'file_format' or returns FileFormat instance intact.

    Args:
        file_format (Union[str, FileFormat]): name of file format or a
            FileFormat instance.

    Raises:
        TypeError: if 'file_format' is neither a str nor FileFormat type.

    Returns:
        FileFormat: appropriate instance.

    """
    if file_format in file_formats:
        return file_formats[file_format]
    elif isinstance(file_format, FileFormat):
        return file_format
    elif inspect.isclass(file_format) and issubclass(file_format, FileFormat):
        return file_format()
    else:
        raise TypeError(f'{file_format} is not a recognized file format')
