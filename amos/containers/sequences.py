"""
sequences: extensible, flexible, lightweight list-like classes
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
    Listing (bunches.Bunch, MutableSequence): drop-in replacement for a python 
        list with additional functionality.
    Hybrid (Listing): iterable with both dict and list interfaces. Stored items
        must be hashable or have a 'name' attribute.
    
ToDo:

        
"""
from __future__ import annotations
from collections.abc import Hashable, Mapping, MutableSequence, Sequence
import copy
import dataclasses
from typing import Any, Optional, Union

from ..observe import report
from ..change import convert
from . import bases

                          
@dataclasses.dataclass # type: ignore
class Listing(bases.Bunch, MutableSequence): # type: ignore
    """Basic amos list replacement.
    
    A Listing differs from an ordinary python list in ways required by 
    inheriting from Bunch: 'add' and 'subset' methods, storing data in 
    'contents', and allowing the '+' operator to join Listings with other lists 
    and Listings) and in 1 other way.
        1) It includes a 'prepend' method for adding one or more items to the
            beginning of the stored list.
    
    The 'add' method attempts to extend 'contents' with the item to be added.
    If this fails, it appends the item to 'contents'.
            
    Args:
        contents (MutableSequence[Any]): items to store in a list. Defaults to 
            an empty list.
        
    """
    contents: MutableSequence[Any] = dataclasses.field(default_factory = list)

    """ Public Methods """

    def add(self, item: Union[Any, Sequence[Any]]) -> None:
        """Tries to extend 'contents' with 'item'. Otherwise, it appends.

        The method will extend all passed sequences, except str types, which it
        will append.
        
        Args:
            item (Union[Any, Sequence[Any]]): item(s) to add to the 'contents' 
                attribute.
                
        """
        if isinstance(item, Sequence) and not isinstance(item, str):
            self.contents.extend(item)
        else:
            self.contents.append(item)
        return

    def insert(self, index: int, item: Any) -> None:
        """Inserts 'item' at 'index' in 'contents'.

        Args:
            index (int): index to insert 'item' at.
            item (Any): object to be inserted.
            
        """
        self.contents.insert(index, item)
        return

    def prepend(self, item: Union[Any, Sequence[Any]]) -> None:
        """Prepends 'item' to 'contents'.

        If 'item' is a non-str sequence, 'prepend' adds its contents to the 
        stored list in the order they appear in 'item'.
        
        Args:
            item (Union[Any, Sequence[Any]]): item(s) to prepend to the 
                'contents' attribute.
                
        """
        if isinstance(item, Sequence) and not isinstance(item, str):
            for thing in reversed(item):
                self.prepend(item = thing)
        else:
            self.insert(0, item)
        return
                   
    def subset(
        self, 
        include: Optional[Union[Sequence[Any], Any]] = None, 
        exclude: Optional[Union[Sequence[Any], Any]] = None) -> Listing:
        """Returns a new instance with a subset of 'contents'.

        This method applies 'include' before 'exclude' if both are passed. If
        'include' is None, all existing keys will be added before 'exclude' is
        applied.
        
        Args:
            include (Optional[Union[Sequence[Any], Any]]): item(s) to include in 
                the new instance. Defaults to None.
            exclude (Optional[Union[Sequence[Any], Any]]): item(s) to exclude in 
                the new instance. Defaults to None.       
        
        Raises:
            ValueError: if 'include' and 'exclude' are both None.
                              
        Returns:
            Listing: with only items from 'include' and no items in 'exclude'.

        """
        if include is None and exclude is None:
            raise ValueError('include or exclude must not be None')
        else:
            if include is None:
                contents = self.contents
            else:
                include = list(convert.iterify(item = include))
                contents = [i for i in self.contents if i in include]
            if exclude is not None:
                exclude = list(convert.iterify(item = exclude))
                contents = [i for i in contents if i not in exclude]
            new_listing = copy.deepcopy(self)
            new_listing.contents = contents
        return new_listing
                       
    """ Dunder Methods """

    def __getitem__(self, index: Any) -> Any:
        """Returns value(s) for 'key' in 'contents'.

        Args:
            index (Any): index to search for in 'contents'.

        Returns:
            Any: item stored in 'contents' at key.

        """
        return self.contents[index]
            
    def __setitem__(self, index: Any, value: Any) -> None:
        """sets 'key' in 'contents' to 'value'.

        Args:
            index (Any): index to set 'value' to in 'contents'.
            value (Any): value to be set at 'key' in 'contents'.

        """
        self.contents[index] = value
        return

    def __delitem__(self, index: Any) -> None:
        """Deletes item at 'key' index in 'contents'.

        Args:
            index (Any): index in 'contents' to delete.

        """
        del self.contents[index]
        return


@dataclasses.dataclass # type: ignore
class Hybrid(Listing):
    """Iterable that has both a dict and list interfaces.
    
    Hybrid combines the functionality and interfaces of python dicts and lists.
    It allows duplicate keys and list-like iteration while supporting the easier
    access methods of dictionaries. In order to support this hybrid approach to
    iterables, Hybrid can only store items that are hashable or have a 'name' 
    attribute or property that contains or returns a hashable value.

    A Hybrid inherits the differences between a Listing and an ordinary python 
    list.
    
    A Hybrid differs from a Listing in 3 significant ways:
        1) It only stores hashable items or objects for which a str name can be
            derived (using the namify function).
        2) Hybrid has an interface of both a dict and a list, but stores a list. 
            Hybrid does this by taking advantage of the 'name' attribute or
            hashability of stored items. A 'name' or hash acts as a key to 
            create the facade of a dict with the items in the stored list 
            serving as values. This allows for duplicate keys for storing items, 
            simpler iteration than a dict, and support for returning multiple 
            matching items. This design comes at the expense of lookup speed. As 
            a result, Hybrid should only be used if a high volume of access 
            calls is not anticipated. Ordinarily, the loss of lookup speed 
            should have negligible effect on overall performance.
        3) Hybrids should not store int types. This ensures that when, for 
            example, a 'hybrid[3]' is called, the item at that index is 
            returned. If int types are stored, that call would create 
            uncertainty as to whether an index or item should be returned. By
            design, int types are assumed to be calls to return the item at that
            index.
        4) When using dict access methods, a list of matches may be returned
            because a Hybrid allows duplicate pseudo-keys to be used.

    Args:
        contents (MutableSequence[Hashable]): items to store that are hashable
            or have a 'name' attribute. Defaults to an empty list.
        default_factory (Optional[Any]): default value to return or default 
            function to call when the 'get' method is used. Defaults to None. 
                          
    """
    contents: MutableSequence[Hashable] = dataclasses.field(
        default_factory = list)
    default_factory: Optional[Any] = None
        
    """ Public Methods """

    def get(self, key: Hashable, default: Optional[Any] = None) -> Any: # type: ignore
        """Returns value in 'contents' or default options.
        
        Args:
            key (Hashable): key for value in 'contents'.
            default (Optional[Any]): default value to return if 'key' is not 
                found in 'contents'.
        
        Raises:
            KeyError: if 'key' is not in the Hybrid and 'default' and the
                'default_factory' attribute are both None.
        
        Returns:
            Any: value matching key in 'contents' or 'default_factory' value. 
            
        """
        try:
            return self[key]
        except (KeyError, TypeError):
            if default is None:
                if self.default_factory is None:
                    raise KeyError(f'{key} is not in the Hybrid')
                else:
                    try:
                        return self.default_factory()
                    except TypeError:
                        return self.default_factory
            else:
                return default

    def items(self) -> tuple[tuple[Hashable, ...], tuple[Any, ...]]:
        """Emulates python dict 'items' method.
        
        Returns:
            tuple[tuple[Hashable, ...], tuple[Any, ...]]: a tuple equivalent to 
                dict.items(). A Hybrid cannot actually create an ItemsView 
                because that would eliminate any duplicate keys, which are 
                permitted by Hybrid.
            
        """
        return tuple(zip(self.keys(), self.values()))

    def keys(self) -> tuple[Hashable, ...]:
        """Emulates python dict 'keys' method.
        
        Returns:
            tuple[Hashable, ...]: a tuple equivalent to dict.keys(). A Hybrid 
                cannot actually create an KeysView because that would eliminate 
                any duplicate keys, which are permitted by Hybrid.
            
        """
        return tuple([convert.namify(item = c) for c in self.contents])

    def setdefault(self, value: Any) -> None: # type: ignore
        """sets default value to return when 'get' method is used.
        
        Args:
            value (Any): default value to return.
            
        """
        self.default_factory = value 
        return

    def update(self, items: Mapping[Any, Any]) -> None:
        """Mimics the dict 'update' method by extending 'contents' with 'items'.
        
        Args:
            items (Mapping[Any, Any]): items to add to the 'contents' attribute. 
                The values of 'items' are added to 'contents' and the keys 
                become the 'name' attributes of those values. As a result, the 
                keys of 'items' are discarded. To mimic dict' update', the 
                passed 'items' values are added to 'contents' by the 'extend' 
                method which adds the values to the end of 'contents'.           
        
        """
        self.extend(list(items.values()))
        return

    def values(self) -> tuple[Any, ...]:
        """Emulates python dict 'values' method.
        
        Returns:
            tuple[Any, ...]: a tuple equivalent to dict.values(). A Hybrid 
                cannot actually create an ValuesView because that would 
                eliminate any duplicate keys, which are permitted by Hybrid.
            
        """
        return tuple(self.contents)

    """ Dunder Methods """

    def __getitem__(self, key: Union[Hashable, int]) -> Any: # type: ignore
        """Returns value(s) for 'key' in 'contents'.
        
        If 'key' is not an int type, this method looks for a matching 'name'
        attribute in the stored instances.
        
        If 'key' is an int type, this method returns the stored item at the
        corresponding index.
        
        If only one match is found, a single item is returned. If more are 
        found, a Hybrid or Hybrid subclass with the matching 'name' attributes 
        is returned.

        Args:
            key (Union[Hashable, int]): name of an item or index to search for 
                in 'contents'.

        Returns:
            Any: value(s) stored in 'contents' that correspond to 'key'. If 
                there is more than one match, the return is a Hybrid or Hybrid 
                subclass with that matching stored items.

        """
        if isinstance(key, int):
            return self.contents[key]
        else:
            
            matches = [
                c for c in self.contents if convert.namify(item = c) == key]
            # matches = []
            # for value in self.contents:
            #     if (
            #         hash(value) == key 
            #         or traits.namify(item = value) == key):
            #         matches.append(value)
            # matches = [
            #     i for i, c in enumerate(self.contents)
            #     if traits.namify(c) == key]
            if len(matches) == 0:
                raise KeyError(f'{key} is not in {self.__class__.__name__}')
            elif len(matches) == 1:
                return matches[0]
            else:
                return matches
            
    def __setitem__(self, key: Union[Any, int], value: Any) -> None:
        """sets 'key' in 'contents' to 'value'.

        Args:
            key (Union[Any, int]): if key isn't an int, it is ignored (since the
                'name' attribute of the value will be acting as the key). In
                such a case, the 'value' is added to the end of 'contents'. If
                key is an int, 'value' is assigned at the that index number in
                'contents'.
            value (Any): value to be paired with 'key' in 'contents'.

        """
        if isinstance(key, int):
            self.contents[key] = value
        else:
            self.add(value)
        return

    def __delitem__(self, key: Union[Any, int]) -> None:
        """Deletes item matching 'key' in 'contents'.

        If 'key' is not an int type, this method looks for a matching 'name'
        attribute in the stored instances and deletes all such items. If 'key'
        is an int type, only the item at that index is deleted.

        Args:
            key (Union[Any, int]): name or index in 'contents' to delete.

        """
        if isinstance(key, int):
            del self.contents[key]
        else:
            self.contents = [
                c for c in self.contents if convert.namify(c) != key]
        return

