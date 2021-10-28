"""
mappings: extensible, flexible, lightweight dict-like classes
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
    Dictionary (Bunch, MutableMapping): bunches's drop-in replacement for a 
        python dict with some added functionality.
    Catalog (Dictionary): wildcard-accepting dict which is primarily intended 
        for storing different options and strategies. It also returns lists of 
        matches if a list of keys is provided.
 
ToDo:

       
"""
from __future__ import annotations
from collections.abc import Hashable, Iterator, Mapping, MutableMapping, Sequence
import copy
import dataclasses
import inspect
from typing import Any, Optional, Type, Union

from ..observe import traits
from ..repair import convert
from . import bunches
                  

_ALL_KEYS: list[Any] = ['all', 'All', ['all'], ['All']]
_DEFAULT_KEYS: list[Any] = [
    'default', 'defaults', 'Default', 'Defaults', ['default'], ['defaults'], 
    ['Default'], ['Defaults']]
_NONE_KEYS: list[Any] = ['none', 'None', ['none'], ['None']]


@dataclasses.dataclass  # type: ignore
class Dictionary(bunches.Bunch, MutableMapping):  # type: ignore
    """Basic bunches dict replacement.
    
    A Dictionary differs from an ordinary python dict in ways inherited from 
    Bunch by requiring 'add' and 'subset' methods, storing data in 'contents', 
    and allowing the '+' operator to join Dictionary instances with other 
    mappings, including Dictionary instances. 
    
    # In addition, it differs in 1 other significant way:
    #     1) When returning 'keys', 'values' and 'items', this class returns them
    #         as tuples instead of KeysView, ValuesView, and ItemsView.
    
    Args:
        contents (MutableMapping[Hashable, Any]): stored dictionary. Defaults 
            to an empty dict.
        default_factory (Optional[Any]): default value to return or default 
            function to call when the 'get' method is used. Defaults to None. 
                          
    """
    contents: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = dict)
    default_factory: Optional[Any] = None

    """ Public Methods """
     
    def add(self, item: Mapping[Hashable, Any], **kwargs: Any) -> None:
        """Adds 'item' to the 'contents' attribute.
        
        Args:
            item (Mapping[Hashable, Any]): items to add to 'contents' attribute.
            kwargs: creates a consistent interface even when subclasses have
                additional parameters.
                
        """
        self.contents.update(item, **kwargs)
        return

    @classmethod
    def fromkeys(
        cls, 
        keys: Sequence[Hashable], 
        value: Any, 
        **kwargs: Any) -> Dictionary:
        """Emulates the 'fromkeys' class method from a python dict.

        Args:
            keys (Sequence[Hashable]): items to be keys in a new Dictionary.
            value (Any): the value to use for all values in a new Dictionary.

        Returns:
            Dictionary: formed from 'keys' and 'value'.
            
        """
        return cls(contents = dict.fromkeys(keys, value), **kwargs)
    
    def get(self, key: Hashable, default: Optional[Any] = None) -> Any: # type: ignore
        """Returns value in 'contents' or default options.
        
        Args:
            key (Hashable): key for value in 'contents'.
            default (Optional[Any]): default value to return if 'key' is not 
                found in 'contents'.
        
        Raises:
            KeyError: if 'key' is not in the Dictionary and 'default' and the
                'default_factory' attribute are both None.
        
        Returns:
            Any: value matching key in 'contents' or 'default_factory' value. 
            
        """
        try:
            return self[key]
        except (KeyError, TypeError):
            if default is None:
                if self.default_factory is None:
                    raise KeyError(f'{key} is not in the Dictionary')
                else:
                    try:
                        return self.default_factory()
                    except TypeError:
                        return self.default_factory
            else:
                return default
                
    # def items(self) -> tuple[tuple[Hashable, Any], ...]: # type: ignore
    #     """Emulates python dict 'items' method.
        
    #     Returns:
    #         tuple[tuple[Hashable], Any]: a tuple equivalent to dict.items(). 
            
    #     """
    #     return tuple(zip(self.keys(), self.values()))

    # def keys(self) -> tuple[Hashable, ...]: # type: ignore
    #     """Returns 'contents' keys as a tuple.
        
    #     Returns:
    #         tuple[Hashable, ...]: a tuple equivalent to dict.keys().
            
    #     """
    #     return tuple(self.contents.keys())

    def setdefault(self, value: Any) -> None: # type: ignore
        """sets default value to return when 'get' method is used.
        
        Args:
            value (Any): default value to return when 'get' is called and the
                'default' parameter to 'get' is None.
            
        """
        self.default_factory = value 
        return
               
    def subset(
        self, 
        include: Optional[Union[Hashable, Sequence[Hashable]]] = None, 
        exclude: Optional[Union[Hashable, Sequence[Hashable]]] = None) -> (
            Dictionary):
        """Returns a new instance with a subset of 'contents'.

        This method applies 'include' before 'exclude' if both are passed. If
        'include' is None, all existing keys will be added before 'exclude' is
        applied.
        
        Args:
            include (Optional[Union[Hashable, Sequence[Hashable]]]): key(s) to 
                include in the new Dictionary instance.
            exclude (Optional[Union[Hashable, Sequence[Hashable]]]): key(s) to 
                exclude in the new Dictionary instance.                
        
        Raises:
            ValueError: if 'include' and 'exclude' are both None.
                  
        Returns:
            Dictionary: with only keys from 'include' and no keys in 'exclude'.

        """
        if include is None and exclude is None:
            raise ValueError('include or exclude must not be None')
        else:
            if include is None:
                contents = self.contents
            else:
                include = list(convert.iterify(item = include)) 
                contents = {k: self.contents[k] for k in include}
            if exclude is not None:
                exclude = list(convert.iterify(item = exclude))
                contents = {
                    k: v for k, v in contents.items() 
                    if k not in exclude}
            new_dictionary = copy.deepcopy(self)
            new_dictionary.contents = contents
        return new_dictionary
      
    # def values(self) -> tuple[Any, ...]: # type: ignore
    #     """Returns 'contents' values as a tuple.
        
    #     Returns:
    #         tuple[Any, ...]: a tuple equivalent to dict.values().
            
    #     """
    #     return tuple(self.contents.values())

    """ Dunder Methods """

    def __getitem__(self, key: Hashable) -> Any:
        """Returns value for 'key' in 'contents'.

        Args:
            key (Hashable): key in 'contents' for which a value is sought.

        Returns:
            Any: value stored in 'contents'.

        """
        return self.contents[key]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """sets 'key' in 'contents' to 'value'.

        Args:
            key (Hashable): key to set in 'contents'.
            value (Any): value to be paired with 'key' in 'contents'.

        """
        self.contents[key] = value
        return

    def __delitem__(self, key: Hashable) -> None:
        """Deletes 'key' in 'contents'.

        Args:
            key (Hashable): key in 'contents' to delete the key/value pair.

        """
        del self.contents[key]
        return


@dataclasses.dataclass  # type: ignore
class Catalog(Dictionary):
    """Wildcard and list-accepting dictionary.

    A Catalog inherits the differences between a Dictionary and an ordinary 
    python dict.

    A Catalog differs from a Dictionary in 5 significant ways:
        1) It recognizes an 'all' key which will return a list of all values
            stored in a Catalog instance.
        2) It recognizes a 'default' key which will return all values matching
            keys listed in the 'default' attribute. 'default' can also be set
            using the 'catalog['default'] = new_default' assignment. If 
            'default' is not passed when the instance is initialized, the 
            initial value of 'default' is 'all'.
        3) It recognizes a 'none' key which will return an empty list.
        4) It supports a list of keys being accessed with the matching values 
            returned. For example, 'catalog[['first_key', 'second_key']]' will 
            return the values for those keys in a list ['first_value',
            'second_value'].
        5) If a single key is sought, a Catalog can either return the stored
            value or a stored value in a list (if 'always_return_list' is
            True). The latter option is available to make iteration easier
            when the iterator assumes a single type will be returned.

    Args:
        contents (Mapping[Hashable, Any]]): stored dictionary. Defaults to an 
            empty dict.
        default_factory (Any): default value to return when the 'get' method is 
            used.
        default (Sequence[Any]]): a list of keys in 'contents' which will be 
            used to return items when 'default' is sought. If not passed, 
            'default' will be set to all keys.
        always_return_list (bool): whether to return a list even when the key 
            passed is not a list or special access key (True) or to return a 
            list only when a list or special access key is used (False). 
            Defaults to False.
                     
    """
    contents: Mapping[Hashable, Any] = dataclasses.field(
        default_factory = dict)
    default_factory: Optional[Any] = None
    default: Optional[Any] = 'all'
    always_return_list: bool = False

    """ Dunder Methods """

    def __getitem__(
        self, 
        key: Union[Hashable, Sequence[Hashable]]) -> Union[Any, Sequence[Any]]:
        """Returns value(s) for 'key' in 'contents'.

        The method searches for 'all', 'default', and 'none' matching wildcard
        options before searching for direct matches in 'contents'.

        Args:
            key (Union[Hashable, Sequence[Hashable]]): key(s) in 'contents'.

        Returns:
            Union[Any, Sequence[Any]]: value(s) stored in 'contents'.

        """
        # Returns a list of all values if the 'all' key is sought.
        if key in _ALL_KEYS:
            return list(self.contents.values())
        # Returns a list of values for keys listed in 'default' attribute.
        elif key in _DEFAULT_KEYS:
            return self[self.default]
        # Returns an empty list if a null value is sought.
        elif key in _NONE_KEYS:
            if self.default_factory is None:
                if self.always_return_list:
                    return []
                else:
                    return None
            else:
                try:
                    return self.default_factory()
                except TypeError:
                    return self.default_factory
        # Returns list of matching values if 'key' is list-like.        
        elif isinstance(key, Sequence) and not isinstance(key, str):
            return [self.contents[k] for k in key if k in self.contents]
        # Returns matching value if key is not a non-str Sequence or wildcard.
        else:
            try:
                if self.always_return_list:
                    return [self.contents[key]]
                else:
                    return self.contents[key]
            except KeyError:
                raise KeyError(f'{key} is not in {self.__class__.__name__}')

    def __setitem__(
        self, 
        key: Union[Hashable, Sequence[Hashable]], 
        value: Union[Any, Sequence[Any]]) -> None:
        """sets 'key' in 'contents' to 'value'.

        Args:
            key (Union[Hashable, Sequence[Hashable]]): key(s) to set in 
                'contents'.
            value (Union[Any, Sequence[Any]]): value(s) to be paired with 'key' 
                in 'contents'.

        """
        try:
            self.contents[key] = value
        except TypeError:
            self.contents.update(dict(zip(key, value))) # type: ignore
        return

    def __delitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> None:
        """Deletes 'key' in 'contents'.

        Args:
            key (Union[Hashable, Sequence[Hashable]]): name(s) of key(s) in 
                'contents' to delete the key/value pair.

        """
        keys = list(convert.iterify(item = key))
        if all(k in self for k in keys):
            self.contents = {
                i: self.contents[i] for i in self.contents if i not in keys}
        else:
            raise KeyError(f'{key} not found in the Catalog')
        return


@dataclasses.dataclass  # type: ignore
class Library(MutableMapping):
    """Stores classes and class instances.
    
    When searching for matches, instances are prioritized over classes.
    
    Args:
        classes (Catalog): a catalog of stored classes. Defaults to any empty
            Catalog.
        instances (Catalog): a catalog of stored class instances. Defaults to an
            empty Catalog.
    
    Attributes:
        maps (list[Catalog]): the ordered mappings to search, as required from
             inheriting from ChainMap.
                 
    """
    classes: Catalog = dataclasses.field(default_factory = dict)
    instances: Catalog = dataclasses.field(default_factory = dict)
        
    """ Public Methods """
    
    def deposit(
        self, 
        item: Union[Type[Any], object],
        name: Optional[Hashable] = None) -> None:
        """Adds 'item' to 'classes' and/or 'instances'.

        If 'item' is a class, it is added to 'classes.' If it is an object, it
        is added to 'instances' and its class is added to 'classes'.
        
        Args:
            item (Union[Type, object]): class or instance to add to the Library
                instance.
            name (Optional[Hashable]): key to use to store 'item'. If not
                passed, a key will be created using the 'get_name' method.
                
        """
        key = name or traits.get_name(item = item)
        if inspect.isclass(item):
            self.classes[key] = item
        elif isinstance(item, object):
            self.instances[key] = item
            self.deposit(item = item.__class__)
        else:
            raise TypeError(f'item must be a class or a class instance')
        return
    
    def remove(self, item: Hashable) -> None:
        """Removes an item from 'instances' or 'classes.'
        
        If 'item' is found in 'instances', it will not also be removed from 
        'classes'.

        Args:
            item (Hashable): key name of item to remove.
            
        Raises:
            KeyError: if 'item' is neither found in 'instances' or 'classes'.

        """
        try:
            del self.instances[item]
        except KeyError:
            try:
                del self.classes[item]
            except KeyError:
                raise KeyError(f'{item} is not found in the Library')
        return    

    def withdraw(
        self, 
        item: Union[Hashable, Sequence[Hashable]], 
        kwargs: Optional[MutableMapping[Hashable, Any]] = None) -> (
            Union[Type[Any], object]):
        """Returns instance or class of first match of 'item' from catalogs.
        
        The method prioritizes the 'instances' catalog over 'classes' and any
        passed names in the order they are listed.
        
        Args:
            item (Union[Hashable, Sequence[Hashable]]): key name(s) of stored 
                item(s) sought.
            kwargs (Optional[MutableMapping[Hashable, Any]]]): keyword arguments 
                to pass to a newly created instance or, if the stored item is 
                already an instance to be manually added as attributes. If not
                passed, the found item will be returned unaltered. Defaults to
                None.
            
        Raises:
            KeyError: if 'item' does not match a key to a stored item in either
                'instances' or 'classes'.
            
        Returns:
            Union[Type[Any], object]: returns a class or instance if 'kwargs' 
                are None, depending upon with Catalog the matching item is 
                found. If 'kwargs' are passed, an instance is always returned.
            
        """
        items = convert.iterify(item)
        item = None
        for key in items:
            for catalog in ['instances', 'classes']:
                try:
                    item = getattr(self, catalog)[key]
                    break
                except KeyError:
                    pass
            if item is not None:
                break
        if item is None:
            raise KeyError(f'No matching item for {item} was found')
        if kwargs is not None:
            if 'item' in item.__annotations__.keys() and 'item' not in kwargs:
                kwargs[item] = items[0]
            if inspect.isclass(item):
                item = item(**kwargs)
            else:
                for key, value in kwargs.items():
                    setattr(item, key, value)  
        return item # type: ignore
    
    """ Dunder Methods """

    def __getitem__(self, key: Hashable) -> Any:
        """Returns value for 'key' in 'contents'.

        Args:
            key (Hashable): key in 'contents' for which a value is sought.

        Returns:
            Any: value stored in 'contents'.

        """
        return self.withdraw(item = key)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """sets 'key' in 'contents' to 'value'.

        Args:
            key (Hashable): key to set in 'contents'.
            value (Any): value to be paired with 'key' in 'contents'.

        """
        self.deposit(item = value, name = key)
        return

    def __delitem__(self, key: Hashable) -> None:
        """Deletes 'key' in 'contents'.

        Args:
            key (Hashable): key in 'contents' to delete the key/value pair.

        """
        self.remove(item = key)
        return

    def __iter__(self) -> Iterator[Any]:
        """Returns iterable of 'contents'.

        Returns:
            Iterator: of 'contents'.

        """
        combined = copy.deepcopy(self.instances)
        return iter(combined.update(self.classes))

    def __len__(self) -> int:
        """Returns length of 'contents'.

        Returns:
            int: length of 'contents'.

        """
        combined = copy.deepcopy(self.instances)
        return len(combined.update(self.classes))
    