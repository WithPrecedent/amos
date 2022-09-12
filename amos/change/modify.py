"""
modify: functions that modify stored data without changing the data type
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
    Adders:
        add_prefix (Callable, dispatcher): adds a str prefix to item.
        add_slots: adds '__slots__' to a dataclass.
        add_suffix (Callable, dispatcher): adds a str suffix to item.
    Dividers:
        cleave (Callable, dispatcher): divides an item into 2 parts based on
            'divider'.
        separate (Callable, dispatcher): divides an item into n+1 parts based on
            'divider'.
    Subtractors:
        deduplicate (Callable, dispatcher): removes duplicate data from an item.
        drop_dunders: drops strings from a list if they start and end with 
            double underscores.
        drop_prefix (Callable, dispatcher): removes a str prefix from an item.
        drop_prefix_from_dict
        drop_prefix_from_list
        drop_prefix_from_set
        drop_prefix_from_str
        drop_prefix_from_tuple
        drop_privates
        drop_substring (Callable, dispatcher): removes a substring from an item.
        drop_suffix (Callable, dispatcher): removes a str suffix from an item.
    Other: 
        capitalify: converts a snake case str to capital case.
        snakify: converts a capital case str to snake case.
        uniquify: returns a unique key for a dict.

ToDo:
    Reintegrate dispatcher from ashworth package once it has been tested.

"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, MutableSequence, Sequence, Set
import dataclasses
import re
from typing import Any, Type


""" Adders """

# @amos.dynamic.dispatcher # type: ignore
def add_prefix(item: Any, prefix: str, divider: str = '') -> Any:
    """Adds 'prefix' to 'item' with 'divider' in between.
    
    Args:
        item (Any): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.

    Raises:
        TypeError: if no registered function supports the type of 'item'.
        
    Returns:
        Any: modified item.

    """
    raise TypeError(f'item is not a supported type for {__name__}')
 
# @add_prefix.register # type: ignore
def add_prefix_to_str(item: str, prefix: str, divider: str = '') -> str:
    """Adds 'prefix' to 'item' with 'divider' in between.
    
    Args:
        item (str): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.

    Returns:
        str: modified str.

    """
    return divider.join([prefix, item])
 
# @add_prefix.register # type: ignore
def add_prefix_to_dict(
    item: Mapping[str, Any],  
    prefix: str, 
    divider: str = '') -> Mapping[str, Any]:
    """Adds 'prefix' to keys in 'item' with 'divider' in between.
    
    Args:
        item (Mapping[str, Any]): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.

    Returns:
        Mapping[str, Any]: modified mapping.

    """
    contents = {
        add_prefix(item = k, prefix = prefix, divider = divider): v 
        for k, v in item.items()}
    if isinstance(item, dict):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore
 
# @add_prefix.register # type: ignore
def add_prefix_to_list(
    item: MutableSequence[str], 
    prefix: str, 
    divider: str = '') -> MutableSequence[str]:
    """Adds 'prefix' to items in 'item' with 'divider' in between.
    
    Args:
        item (MutableSequence[str]): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.

    Returns:
        Any: modified sequence.

    """
    contents = [
        add_prefix(item = i, prefix = prefix, divider = divider) for i in item]
    if isinstance(item, list):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore
 
# @add_prefix.register # type: ignore
def add_prefix_to_set(
    item: Set[str], 
    prefix: str, 
    divider: str = '') -> Set[str]:
    """Adds 'prefix' to items in 'item' with 'divider' in between.
    
    Args:
        item (Set[str]): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.

    Returns:
        Set[str]: modified set.

    """
    contents = {
        add_prefix(item = i, prefix = prefix, divider = divider) for i in item}
    if isinstance(item, set):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore

# @add_prefix.register # type: ignore
def add_prefix_to_tuple(
    item: tuple[str, ...], 
    prefix: str, 
    divider: str = '') -> tuple[str, ...]:
    """Adds 'prefix' to items in 'item' with 'divider' in between.
    
    Args:
        item (tuple[str, ...]): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.

    Raises:
        TypeError: if no registered function supports the type of 'item'.
        
    Returns:
        tuple[str, ...]: modified tuple.

    """
    return tuple(
        [add_prefix(item = i, prefix = prefix, divider = divider) 
         for i in item])

def add_slots(item: Type[Any]) -> Type[Any]:
    """Adds slots to dataclass with default values.
    
    Derived from code here: 
    https://gitquirks.com/ericvsmith/dataclasses/blob/master/dataclass_tools.py
    
    Args:
        item (Type[Any]): dataclass to add slots to.
        
    Raises:
        TypeError: if '__slots__' is already in item.
        
    Returns:
        Type[Any]: class with '__slots__' added.
        
    """
    if '__slots__' in item.__dict__:
        raise TypeError(f'{item.__name__} already contains __slots__')
    else:
        item_dict = dict(item.__dict__)
        field_names = tuple(f.name for f in dataclasses.fields(item))
        item_dict['__slots__'] = field_names
        for field_name in field_names:
            item_dict.pop(field_name, None)
        item_dict.pop('__dict__', None)
        qualname = getattr(item, '__qualname__', None)
        item = type(item)(item.__name__, item.__bases__, item_dict) # type: ignore
        if qualname is not None:
            item.__qualname__ = qualname
    return item

# @amos.dynamic.dispatcher # type: ignore 
def add_suffix(item: Any, suffix: str, divider: str = '') -> Any:
    """Adds 'suffix' to 'item' with 'divider' in between.
    
    Args:
        item (Any): item to be modified.
        suffix (str): suffix to be added to 'item'.
        divider (str): str to add between 'item' and 'suffix'. Defaults to '',
            which means no divider will be added.

    Raises:
        TypeError: if no registered function supports the type of 'item'.
        
    Returns:
        Any: modified item.

    """
    raise TypeError(f'item is not a supported type for {__name__}')
 
# @add_suffix.register # type: ignore
def add_suffix_to_str(item: str, suffix: str, divider: str = '') -> str:
    """Adds 'suffix' to 'item' with 'divider' in between.
    
    Args:
        item (str): item to be modified.
        suffix (str): suffix to be added to 'item'.
        divider (str): str to add between 'item' and 'suffix'. Defaults to '',
            which means no divider will be added.

    Returns:
        str: modified item.

    """
    return divider.join([item, suffix])
 
# @add_suffix.register # type: ignore
def add_suffix_to_dict(
    item: Mapping[str, Any], 
    suffix: str, 
    divider: str = '') -> Mapping[str, Any]:
    """Adds 'suffix' to keys in 'item' with 'divider' in between.
    
    Args:
        item (Mapping[str, Any]): item to be modified.
        suffix (str): suffix to be added to 'item'.
        divider (str): str to add between 'item' and 'suffix'. Defaults to '',
            which means no divider will be added.

    Returns:
        Mapping[str, Any]: modified mapping.

    """
    contents = {
        add_suffix(item = k, suffix = suffix, divider = divider): v 
        for k, v in item.items()}
    if isinstance(item, dict):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore
 
# @add_suffix.register # type: ignore
def add_suffix_to_list(
    item: MutableSequence[str], 
    suffix: str, 
    divider: str = '') -> MutableSequence[str]:
    """Adds 'suffix' to items in 'item' with 'divider' in between.
    
    Args:
        item (MutableSequence[str]): item to be modified.
        suffix (str): suffix to be added to 'item'.
        divider (str): str to add between 'item' and 'suffix'. Defaults to '',
            which means no divider will be added.

    Returns:
        MutableSequence[str]: modified sequence.

    """
    contents = [
        add_suffix(item = i, suffix = suffix, divider = divider) for i in item]
    if isinstance(item, list):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore
 
# @add_suffix.register # type: ignore
def add_suffix_to_set(
    item: Set[str], 
    suffix: str, 
    divider: str = '') -> Set[str]:
    """Adds 'suffix' to items in 'item' with 'divider' in between.
    
    Args:
        item (Set[str]): item to be modified.
        suffix (str): suffix to be added to 'item'.
        divider (str): str to add between 'item' and 'suffix'. Defaults to '',
            which means no divider will be added.

    Returns:
        Set[str]: modified set.

    """
    contents = {add_suffix(item = i, suffix = suffix, divider = divider) 
                for i in item}
    if isinstance(item, set):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore

# @add_suffix.register # type: ignore
def add_suffix_to_tuple(
    item: tuple[str, ...], 
    suffix: str, 
    divider: str = '') -> tuple[str, ...]:
    """Adds 'suffix' to items in 'item' with 'divider' in between.
    
    Args:
        item (tuple[str, ...]): item to be modified.
        suffix (str): suffix to be added to 'item'.
        divider (str): str to add between 'item' and 'suffix'. Defaults to '',
            which means no divider will be added.

    Returns:
        tuple[str, ...]: modified tuple.

    """
    return tuple(
        [add_suffix(item = i, suffix = suffix, divider = divider) 
         for i in item])

""" Dividers """

# @amos.dynamic.dispatcher # type: ignore
def cleave(
    item: Any, 
    divider: Any,
    return_last: bool = True,
    raise_error: bool = False) -> tuple[Any, Any]:
    """Divides 'item' into 2 parts based on 'divider'.

    Args:
        item (Any): item to be divided.
        divider (Any): item to divide 'item' upon.
        return_last (bool): whether to split 'item' upon the first (False) or
            last appearance of 'divider'.
        raise_error (bool): whether to raise an error if 'divider' is not in 
            'item' or to return a tuple containing 'item' twice.

    Raises:
        TypeError: if no registered function supports the type of 'item'. 
        
    Returns:
        tuple[Any, Any]: parts of 'item' on either side of 'divider' unless
            'divider' is not in 'item'.
        
    """
    raise TypeError(f'item is not a supported type for {__name__}')

# @cleave.register # type: ignore
def cleave_str(
    item: str, 
    divider: str = '_',
    return_last: bool = True,
    raise_error: bool = False) -> tuple[str, str]:
    """Divides 'item' into 2 parts based on 'divider'.

    Args:
        item (str): item to be divided.
        divider (str): item to divide 'item' upon.
        return_last (bool): whether to split 'item' upon the first (False) or
            last appearance of 'divider'.
        raise_error (bool): whether to raise an error if 'divider' is not in 
            'item' or to return a tuple containing 'item' twice.

    Raises:
        ValueError: if 'divider' is not in 'item' and 'raise_error' is True.
        
    Returns:
        tuple[str, str]: parts of 'item' on either side of 'divider' unless
            'divider' is not in 'item'.
        
    """
    if divider in item:
        if return_last:
            suffix = item.split(divider)[-1]
        else:
            suffix = item.split(divider)[0]
        prefix = item[:-len(suffix) - 1]
    elif raise_error:
        raise ValueError(f'{divider} is not in {item}')
    else:
        prefix = suffix = item
    return prefix, suffix

# @amos.dynamic.dispatcher # type: ignore
def separate(
    item: Any, 
    divider: Any,
    raise_error: bool = False) -> tuple[Any, ...]:
    """Divides 'item' into n+1 parts based on 'divider'.

    Args:
        item (Any): item to be divided.
        divider (Any): item to divide 'item' upon.
        raise_error (bool): whether to raise an error if 'divider' is not in 
            'item' or to return a tuple containing 'item' twice.

    Raises:
        TypeError: if no registered function supports the type of 'item'. 
        
    Returns:
        list[Any, ...]: parts of 'item' on either side of 'divider' unless
            'divider' is not in 'item'.
        
    """
    raise TypeError(f'item is not a supported type for {__name__}')

# @separate.register # type: ignore
def separate_str(
    item: str, 
    divider: str = '_',
    raise_error: bool = False) -> list[str]:
    """Divides 'item' into n+1 parts based on 'divider'.

    Args:
        item (str): item to be divided.
        divider (str): item to divide 'item' upon.
        raise_error (bool): whether to raise an error if 'divider' is not in 
            'item' or to return a tuple containing 'item' twice.

    Raises:
        ValueError: if 'divider' is not in 'item' and 'raise_error' is True.
        
    Returns:
        list[str]: parts of 'item' on either side of 'divider' unless 'divider' 
            is not in 'item'.
        
    """
    if divider in item:
        return item.split(divider)
    elif raise_error:
        raise ValueError(f'{divider} is not in {item}')
    else:
        return [item]
 
""" Subtractors """

# @amos.dynamic.dispatcher # type: ignore
def deduplicate(item: Any) -> Any:
    """Deduplicates contents of 'item.
    
    Args:
        item (Any): item to deduplicate.

    Raises:
        TypeError: if no registered function supports the type of 'item'.     
        
    Returns:
        Any: deduplicated item.
        
    """
    raise TypeError(f'item is not a supported type for {__name__}')

# @deduplicate.register # type: ignore
def deduplicate_list(item: MutableSequence[Any]) -> MutableSequence[Any]:
    """Deduplicates contents of 'item.
    
    Args:
        item (MutableSequence[Any]): item to deduplicate.

    Returns:
        MutableSequence[Any]: deduplicated item.
        
    """
    contents = list(dict.fromkeys(item))
    if isinstance(item, list):
        return contents
    else:
        vessel = item.__class__(contents) # type: ignore
        return vessel(contents) # type: ignore

# @deduplicate.register # type: ignore
def deduplicate_tuple(item: tuple[Any, ...]) -> tuple[Any, ...]:
    """Deduplicates contents of 'item.
    
    Args:
        item (tuple[Any, ...]): item to deduplicate.

    Returns:
        tuple[Any, ...]: deduplicated item.
        
    """
    return tuple(list(dict.fromkeys(item)))

def drop_dunders(item: list[Any]) -> list[Any]:
    """Drops items in 'item' with names beginning with an underscore.

    Args:
        item (list[Any]): attributes, methods, and properties of a class.

    Returns:
        list[Any]: attributes, methods, and properties that do not start with an
            underscore.
        
    """
    if len(item) > 0 and hasattr(item[0], '__name__'):
        return [
            i for i in item 
            if not i.__name__.startswith('__') 
            and not i.__name__.endswith('__')]
    else:
        return [
            i for i in item if not i.startswith('__') and not i.endswith('__')]
    
# @amos.dynamic.dispatcher # type: ignore
def drop_prefix(item: Any, prefix: str, divider: str = '') -> Any:
    """Drops 'prefix' from 'item' with 'divider' in between.
    
    Args:
        item (Any): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.
            
    Raises:
        TypeError: if no registered function supports the type of 'item'.
        
    Returns:
        Any: modified item.

    """
    raise TypeError(f'item is not a supported type for {__name__}')

# @drop_prefix.register # type: ignore
def drop_prefix_from_str(item: str, prefix: str, divider: str = '') -> str:
    """Drops 'prefix' from 'item' with 'divider' in between.
    
    Args:
        item (str): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.
 
    Returns:
        str: modified str.

    """
    prefix = ''.join([prefix, divider])
    if item.startswith(prefix):
        return item[len(prefix):]
    else:
        return item

# @drop_prefix.register # type: ignore
def drop_prefix_from_dict(
    item: Mapping[str, Any], 
    prefix: str, 
    divider: str = '') -> Mapping[str, Any]:
    """Drops 'prefix' from keys in 'item' with 'divider' in between.
    
    Args:
        item (Mapping[str, Any]): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.
 
    Returns:
        Mapping[str, Any]: modified mapping.

    """
    contents = {
        drop_prefix(item = k, prefix = prefix, divider = divider): v
        for k, v in item.items()}
    if isinstance(item, dict):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore

# @drop_prefix.register # type: ignore
def drop_prefix_from_list(
    item: MutableSequence[str], 
    prefix: str, 
    divider: str = '') -> MutableSequence[str]:
    """Drops 'prefix' from items in 'item' with 'divider' in between.
    
    Args:
        item (MutableSequence[str]): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.
 
    Returns:
        MutableSequence[str]: modified sequence.

    """
    contents = [
        drop_prefix(item = i, prefix = prefix, divider = divider) for i in item] 
    if isinstance(item, list):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore

# @drop_prefix.register # type: ignore
def drop_prefix_from_set(
    item: Set[str], 
    prefix: str, 
    divider: str = '') -> Set[str]:
    """Drops 'prefix' from items in 'item' with 'divider' in between.
    
    Args:
        item (Set[str]): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.
 
    Returns:
        Set[str]: modified set.

    """
    contents = {
        drop_prefix(item = i, prefix = prefix, divider = divider) for i in item}   
    if isinstance(item, set):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore # type: ignore  

# @drop_prefix.register # type: ignore
def drop_prefix_from_tuple(
    item: tuple[str, ...], 
    prefix: str, 
    divider: str = '') -> tuple[str, ...]:
    """Drops 'prefix' from items in 'item' with 'divider' in between.
    
    Args:
        item (tuple[str, ...]): item to be modified.
        prefix (str): prefix to be added to 'item'.
        divider (str): str to add between 'item' and 'prefix'. Defaults to '',
            which means no divider will be added.
 
    Returns:
        tuple[str, ...]: modified tuple.

    """
    return tuple(
        [drop_prefix(item = i, prefix = prefix, divider = divider) 
         for i in item])       

def drop_privates(item: list[Any]) -> list[Any]:
    """Drops items in 'item' with names beginning with an underscore.

    Args:
        item (list[Any]): attributes, methods, and properties of a class.

    Returns:
        list[Any]: attributes, methods, and properties that do not start with an
            underscore.
        
    """
    if len(item) > 0 and hasattr(item[0], '__name__'):
        return [i for i in item if not i.__name__.startswith('_')]
    else:
        return [i for i in item if not i.startswith('_')]
              
# @amos.dynamic.dispatcher # type: ignore
def drop_substring(item: Any, substring: str) -> Any:
    """Drops 'substring' from 'item' with a possible 'divider' in between.
    
    Args:
        item (Any): item to be modified.
        substring (str): substring to be added to 'item'.
            
    Raises:
        TypeError: if no registered function supports the type of 'item'.
        
    Returns:
        Any: modified item.

    """
    raise TypeError(f'item is not a supported type for {__name__}')

# @drop_substring.register # type: ignore
def drop_substring_from_str(item: str, substring: str) -> str:
    """Drops 'substring' from 'item'.
    
    Args:
        item (str): item to be modified.
        substring (str): substring to be added to 'item'.

    Returns:
        str: modified str.

    """
    if substring in item:
        return item.replace(substring, '')
    else:
        return item

# @drop_substring.register # type: ignore
def drop_substring_from_dict(
    item: Mapping[str, Any], 
    substring: str) -> Mapping[str, Any]:
    """Drops 'substring' from keys in 'item'.
    
    Args:
        item (Mapping[str, Any]): item to be modified.
        substring (str): substring to be added to 'item'.

    Returns:
        Mapping[str, Any]: modified mapping.

    """
    contents = {
        drop_substring(item = k, substring = substring): v
        for k, v in item.items()}
    if isinstance(item, dict):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore

# @drop_substring.register # type: ignore
def drop_substring_from_list(
    item: MutableSequence[str], 
    substring: str) -> MutableSequence[str]:
    """Drops 'substring' from items in 'item'.
    
    Args:
        item (MutableSequence[str]): item to be modified.
        substring (str): substring to be added to 'item'.

    Returns:
        MutableSequence[str]: modified sequence.

    """
    contents = [drop_substring(item = i, substring = substring) for i in item] 
    if isinstance(item, list):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore

# @drop_substring.register # type: ignore
def drop_substring_from_set(item: Set[str], substring: str) -> Set[str]:
    """Drops 'substring' from items in 'item'.
    
    Args:
        item (Set[str]): item to be modified.
        substring (str): substring to be added to 'item'.

    Returns:
        Set[str]: modified set.

    """
    contents = {drop_substring(item = i, substring = substring) for i in item}   
    if isinstance(item, set):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore # type: ignore  

# @drop_substring.register # type: ignore 
def drop_substring_from_tuple(
    item: tuple[str, ...], 
    substring: str) -> tuple[str, ...]:
    """Drops 'substring' from items in 'item'.
    
    Args:
        item (tuple[str, ...]): item to be modified.
        substring (str): substring to be added to 'item'.

    Returns:
        tuple[str, ...]: modified tuple.

    """
    return tuple(
        [drop_substring(item = i, substring = substring) for i in item])    
     
# @amos.dynamic.dispatcher # type: ignore
def drop_suffix(item: Any, suffix: str, divider: str = '') -> Any:
    """Drops 'suffix' from 'item' with 'divider' in between.
    
    Args:
        item (Any): item to be modified.
        suffix (str): suffix to be added to 'item'.

    Raises:
        TypeError: if no registered function supports the type of 'item'.
        
    Returns:
        Any: modified item.

    """
    raise TypeError(f'item is not a supported type for {__name__}')

# @drop_suffix.register # type: ignore
def drop_suffix_from_str(item: str, suffix: str, divider: str = '') -> str:
    """Drops 'suffix' from 'item' with 'divider' in between.
    
    Args:
        item (str): item to be modified.
        suffix (str): suffix to be added to 'item'.

    Returns:
        str: modified str.

    """
    suffix = ''.join([suffix, divider])
    if item.endswith(suffix):
        return item[:len(suffix)]
    else:
        return item

# drop_suffix.register # type: ignore
def drop_suffix_from_dict(
    item: Mapping[str, Any], 
    suffix: str, 
    divider: str = '') -> Mapping[str, Any]:
    """Drops 'suffix' from keys in 'item' with 'divider' in between.
    
    Args:
        item (Mapping[str, Any]): item to be modified.
        suffix (str): suffix to be added to 'item'.

    Returns:
        Mapping[str, Any]: modified mapping.

    """
    contents = {
        drop_suffix(item = k, suffix = suffix, divider = divider): v 
        for k, v in item.items()}
    if isinstance(item, dict):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore

# @drop_suffix.register # type: ignore
def drop_suffix_from_list(
    item: MutableSequence[str], 
    suffix: str, 
    divider: str = '') -> MutableSequence[str]:
    """Drops 'suffix' from items in 'item' with 'divider' in between.
    
    Args:
        item (MutableSequence[str]): item to be modified.
        suffix (str): suffix to be added to 'item'.

    Returns:
        MutableSequence[str]: modified sequence.

    """
    contents = [
        drop_suffix(item = i, suffix = suffix, divider = divider) for i in item]
    if isinstance(item, list):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore

# @drop_suffix.register # type: ignore
def drop_suffix_from_set(
    item: Set[str], 
    suffix: str, 
    divider: str = '') -> Set[str]:
    """Drops 'suffix' from items in 'item' with 'divider' in between.
    
    Args:
        item (Set[str]): item to be modified.
        suffix (str): suffix to be added to 'item'.

    Returns:
        Set[str]: modified set.

    """
    contents = {
        drop_suffix(item = i, suffix = suffix, divider = divider) for i in item}      
    if isinstance(item, set):
        return contents
    else:
        vessel = item.__class__
        return vessel(contents) # type: ignore # type: ignore  

# @drop_suffix.register # type: ignore
def drop_suffix_from_tuple(
    item: tuple[str, ...], 
    suffix: str, 
    divider: str = '') -> tuple[str, ...]:
    """Drops 'suffix' from items in 'item' with 'divider' in between.
    
    Args:
        item (tuple[str, ...]): item to be modified.
        suffix (str): suffix to be added to 'item'.

    Returns:
        tuple[str, ...]: modified tuple.

    """
    return tuple(
        [drop_suffix(item = i, suffix = suffix, divider = divider) 
         for i in item])        

""" Other Modifiers """

def capitalify(item: str) -> str:
    """Converts a snake case str to capital case.

    Args:
        item (str): str to convert.

    Returns:
        str: 'item' converted to capital case.

    """
    return item.replace('_', ' ').title().replace(' ', '')

def snakify(item: str) -> str:
    """Converts a capitalized str to snake case.

    Args:
        item (str): str to convert.

    Returns:
        str: 'item' converted to snake case.

    """
    item = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', item)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', item).lower()

def uniquify(key: str, dictionary: Mapping[Hashable, Any]) -> str:
    """Creates a unique key name to avoid overwriting an item in 'dictionary'.
    
    The function is 1-indexed so that the first attempt to avoid a duplicate
    will be: "old_name2".

    Args:
        key (str): name of key to test.
        dictionary (Mapping[Hashable, Any]): dict for which a unique key name
            is sought.

    Returns:
        str: unique key name for 'dictionary'.
        
    """
    if key not in dictionary:
        return key
    else:
        counter = 1
        while True:
            counter += 1
            if counter > 2:
                name = name.removesuffix(str(counter - 1))
            name = ''.join([key, str(counter)])
            if name not in dictionary:
                return name 