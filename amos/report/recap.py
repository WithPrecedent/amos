"""
recap: functions for better representing python objects as strings
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
    Representation (object): data for a data type's representation.
    facades (Dict): dictionary of different supported types with Representation
        instances as values.
    beautify (Callable): provides a pretty str summary for an object. The
        function uses the 'LINE_BREAK' and 'INDENT' module-level items for
        the values for new lines and length of an indentation.
    beautify_dict (Callable): returns a beautiful string repreentation of a
        dict or dict-like object.
    beautify_object (Callable): returns a beautiful string repreentation of a
        class instance and its attributes.
    beautify_list (Callable): returns a beautiful string repreentation of a
        list, set, tuple, list-like, set-like, or tuple-like object.
    beautify_string (Callable): returns a beautiful string repreentation of a
        str.
    _get_indent (Callable): determines the appropriate indentation for a 
        beautiful str.
    _classify_facade (Callable): called by 'beautify' to determine the 
        appropriate function to beautify the passed 'item'.
         
ToDo:
    Clean up and add DocStrings
    Add a textwrap option when VERTICAL is False.
    
"""
from __future__ import annotations
from collections.abc import (
    Hashable, Iterable, Mapping, MutableMapping, MutableSequence, Sequence)
import dataclasses
import inspect
from types import FunctionType
from typing import Any, Optional, Type, Union

from ..observe import traits
from ..repair import modify


LINE_BREAK: str = '\n'
WHITESPACE: str = ' '
TAB: int = 3
INDENT: str = WHITESPACE * TAB
MAX_WIDTH: int = 40
MAX_LENGTH: int = 20
INCOMPLETE: str = '...'
VERTICAL: bool = True


def get_name(item: Any, default: Optional[str] = None) -> Optional[str]:
    """Returns str name representation of 'item'.
    
    Args:
        item (Any): item to determine a str name.
        default(Optional[str]): default name to return if other methods at name
            creation fail.

    Returns:
        str: a name representation of 'item.'
        
    """        
    if isinstance(item, str):
        return item
    else:
        if hasattr(item, 'name') and isinstance(item.name, str):
            return item.name
        else:
            try:
                return snakify(item.__name__) # type: ignore
            except AttributeError:
                if item.__class__.__name__ is not None:
                    return snakify( # type: ignore
                        item.__class__.__name__) 
                else:
                    return default


@dataclasses.dataclass
class Representation(object):
    """Contains formating information for different data types.
    
    Args:
        name (str): name of data type to be used in the str returned by 
            'beautify'.
        method (FunctionType): the function to use to beautify the particular
            data type.
        start (str): starting bracket for listing the contents of the data type.
            Defaults to ''.
        end (str): ending bracket for listing the contents of the data type.
            Defaults to ''.           
    
    """
    name: str
    method: FunctionType
    start: str = ''
    end: str = ''

""" Public Functions"""
    
def beautify(
    item: Any, 
    offsets: int = 1, 
    package: Optional[str] = None,
    exclude: Optional[MutableSequence[str]] = None,
    include_private: bool = False) -> str:
    """Returns a beautiful string representation of 'item'.

    Args:
        item (Any): item to be represented.
        offsets (int): number of tabs of whitespace to put before the str
            representation. Defaults to 1.
        package (str): name of associated package of 'item'. 'package' is only
            used if 'item' is an object. Defaults to None.
        exclude (MutableSequence[str]): if 'item' is an object, the names of
            attributes to exclude from the str representation. Defaults to None.
        include_private (bool): whether to include attributes with a single 
            leading underscore. Defaults to False.

    Returns:
        str: beautiful str representation of 'item'.
        
    """
    facade = _classify_facade(item = item)
    if facade is None:
        indent = _get_indent(offsets = offsets)
        summary = f'{indent}None'
    else:
        kwargs = {'item': item, 'facade': facade, 'offsets': offsets}
        if facade.name == 'object':
            exclude = exclude or []
            kwargs.update(
                {'package': package, 
                 'exclude': exclude,
                 'include_private': include_private})
        summary = facade.method(**kwargs)
    return f'{LINE_BREAK}{summary}'
   
def beautify_dict(
    item: Mapping[Hashable, Any], 
    facade: Union[Representation, Type[Any]], 
    offsets: int) -> str:
    """Returns a beautiful str representation of a dict-like 'item'.

    Args:
        item (Mapping[Hashable, Any]): item to be represented.
        facade (Union[Representation, Type[Any]]): representation for item or 
            a type matching a key in the 'facades' dict.
        offsets (int): number of tabs of whitespace to put before the str
            representation.

    Returns:
        str: a beautiful representation of 'item'.
        
    """
    if not isinstance(facade, Representation):
        facade = facades[facade]
    indent = _get_indent(offsets = offsets)
    inner = _get_indent(offsets = offsets, extra = TAB)
    summary = [f'{indent}{facade.name}: {facade.start}{LINE_BREAK}']
    length = len(item)
    for i, (key, value) in enumerate(item.items()):
        if i == MAX_LENGTH:
            summary.append(f'{inner}{INCOMPLETE}, {facade.end}{LINE_BREAK}')
            break
        else:
            summary.append(f'{inner}{key}: {value}')
            if i + 1 == length:
                summary.append(f'{facade.end}')
            else:
                summary.append(f',')
            summary.append(f'{LINE_BREAK}')
    return ''.join(summary)

def beautify_object(
    item: object, 
    facade: Union[Representation, Type[Any]], 
    offsets: int,
    package: Optional[str] = None,
    exclude: MutableSequence[str] = None,
    include_private: bool = False) -> str:
    """Returns a beautiful str representation of a class instance.

    Args:
        item (object): item to be represented.
        offsets (int): number of tabs of whitespace to put before the str
            representation. Defaults to 1.
        package (str): name of associated package of 'item'. 'package' is only
            used if 'item' is an object. Defaults to None.
        exclude (MutableSequence[str]): if 'item' is an object, the names of
            attributes to exclude from the str representation. Defaults to None.
        include_private (bool): whether to include attributes with a single 
            leading underscore. Defaults to False.

    Returns:
        str: a beautiful representation of 'item'.
        
    """
    if not isinstance(facade, Representation):
        facade = facades[facade]
    if package is None:
        module = inspect.getmodule(item)
        if hasattr(module, '__package__'):
            package = module.__package__
    if facade.name == 'object':
        name = traits.get_name(item = item)
    else:
        name = ''
    base = modify.snakify(item.__class__.__name__)
    indent = _get_indent(offsets = offsets)
    inner = _get_indent(offsets = offsets, extra = TAB)
    summary = [f'{indent}']
    if name and base and package:
        if name == base:
            summary.append(f'{package} {name}: {LINE_BREAK}')
        else:
            summary.append(f'{name}, ({package} {base}): {LINE_BREAK}')
    else:
        if name == base:
            summary.append(f'{name}: {LINE_BREAK}')
        else:
            summary.append(f'{name}, ({base}): {LINE_BREAK}')  
    if include_private:
        attributes = [a for a in item.__dict__.keys() if not a.startswith('__')]
    else:
        attributes = [a for a in item.__dict__.keys() if not a.startswith('_')]
    attributes = [a for a in attributes if a not in exclude]
    inner_offsets = offsets + 2
    for attribute in attributes:
        contents = getattr(item, attribute)
        summary.append(f'{inner}{attribute}: {facade.start}')
        summary.append(beautify(item = contents, offsets = inner_offsets))
    return ''.join(summary)

def beautify_list(
    item: Union[MutableSequence[Any], set[Any], tuple[Any, ...]], 
    facade: Union[Representation, Type[Any]], 
    offsets: int) -> str:
    """Returns a beautiful string representation of a list-like 'item'.

    Args:
        item (Union[MutableSequence, set, tuple]): the list, set, tuple, or 
            similar object to return a str representation for.
        facade (Union[Representation, Type]): 
        offsets (int): [description]

    Returns:
        str: [description]
    """
    if not isinstance(facade, Representation):
        facade = facades[facade]
    indent = _get_indent(offsets = offsets)
    inner = _get_indent(offsets = offsets, extra = TAB)
    summary = [f'{indent}{facade.name}: {facade.start}{LINE_BREAK}']
    length = len(item)
    for i, sub_item in enumerate(item):
        if i == MAX_LENGTH:
            summary.append(f'{inner}{INCOMPLETE}, {facade.end}{LINE_BREAK}')
            break
        else:
            summary.append(f'{inner}{str(sub_item)}')
            if i + 1 == length:
                summary.append(f'{facade.end}')
            else:
                summary.append(f',')
            summary.append(f'{LINE_BREAK}')
    return ''.join(summary)

def beautify_string(
    item: MutableSequence[Any], 
    facade: Union[Representation, Type[Any]], 
    offsets: int) -> str:
    """[summary]

    Args:
        item (str): [description]
        offsets (int): [description]

    Returns:
        str: [description]
    """
    if not isinstance(facade, Representation):
        facade = facades[facade]
    indent = _get_indent(offsets = offsets)
    return f'{indent}{facade.name}: {facade.start}{item}{facade.end}'

""" Private Functions """

def _get_indent(offsets: int, extra: int = 0) -> str:
    """[summary]

    Args:
        offsets (int): [description]
        extra (int, optional): [description]. Defaults to 0.

    Returns:
        str: [description]
    """
    return offsets * INDENT + extra * WHITESPACE

def _classify_facade(item: Any) -> Representation:
    """[summary]

    Args:
        item (Any): [description]

    Returns:
        str: [description]
    """
    if item is None:
        return None
    else:
        for facade, data in facades.items():
            if isinstance(item, facade):
                return data
    return facades[str]

   
""" Module Level Attributes """

facades: dict[str, Representation] = {}
facades[str] = Representation(
    name = 'string',
    method = beautify_string,
    start = '',
    end = '')
facades[MutableMapping] = Representation(
    name = 'dictionary',
    method = beautify_dict,
    start = '{',
    end = '}')
facades[MutableSequence] = Representation(
    name = 'list',
    method = beautify_list,
    start = '[',
    end = ']')
facades[Sequence] = Representation(
    name = 'tuple',
    method = beautify_list,
    start = '(',
    end = ')')
facades[set] = Representation(
    name = 'set',
    method = beautify_list,
    start = '{',
    end = '}')
facades[object] = Representation(
    name = 'object', 
    method = beautify_object,
    start = '',
    end = '')

       
# def _get_textwrapper() -> textwrap.TextWrapper:
#     """[summary]

#     Returns:
#         textwrap.TextWrapper: [description]
#     """
#     return textwrap.TextWrapper(
#         width = MAX_WIDTH,
#         tabsize = len(INDENT),
#         replace_whitespace = False,
#         drop_whitespace = False,
#         max_lines = MAX_LENGTH,
#         placeholder = '...')