"""
traits: tools for examining classes, instances, and other python objects
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
    contains
    contains_dict
    contains_list
    contains_set
    contains_tuple
    parallel_contains
    serial_contains
    get_annotations
    get_attributes
    get_contents
    get_contents_types
    get_methods
    get_name
    get_properties
    get_signatures
    get_types
    get_variables
    has_attributes
    has_methods
    has_properties
    has_signatures
    has_traits
    has_types
    has_types_dict
    has_types_list
    has_types_sequence
    is_class_attribute
    is_container
    is_function
    is_iterable
    is_method
    is_nested
    is_sequence
    name_attributes
    name_methods
    name_parameters
    name_properties
    name_variables
        
ToDo:
    Adding parsing functionlity to signature-related functions to find
        equivalence when one signature has subtypes of the other signature
        (e.g., one type annotation is 'dict' and the other is 'MutableMapping').
        It might be necessary to create a separate Signature-like class to 
        implement this functionality. This includes fixing or abandoning 
        'has_annotations' due to issues matching type annotations.
    Add support for nagata Kinds once that system is complete.
    Add support for types (using type annotations) in the 'contains' function so
        that 'contains' can be applied to classes and not just instances.
    Add 'dispatcher' framework to 'contains' once the dispatcher framework is
        completed in the 'bobbie' package and the Kind system is completed in
        the nagata package. This should replace existing usages of python's
        singledispatch, which doesn't propertly deal with subtypes.

"""
from __future__ import annotations
from collections.abc import (
    Container, Hashable, Iterable, Mapping, MutableSequence, Sequence, Set)
import functools
import inspect
import types
from typing import Any, Optional, Type, Union

from . import utilities


@functools.singledispatch
def contains(
    item: object,
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether 'item' contains the type(s) in 'contents'.

    Args:
        item (object): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    raise TypeError(f'item {item} is not supported by {__name__}')

@contains.register(Mapping)    
def dict_contains(
    item: Mapping[Hashable, Any], 
    contents: tuple[Union[Type[Any], tuple[Type[Any], ...]],
                    Union[Type[Any], tuple[Type[Any], ...]]]) -> bool:
    """Returns whether dict 'item' contains the type(s) in 'contents'.

    Args:
        item (Mapping[Hashable, Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return (
        serial_contains(item = item.keys(), contents = contents[0])
        and serial_contains(item = item.values(), contents = contents[1]))

@contains.register(MutableSequence)   
def list_contains(
    item: MutableSequence[Any],
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether list 'item' contains the type(s) in 'contents'.

    Args:
        item (MutableSequence[Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return serial_contains(item = item, contents = contents)

@contains.register(Set)   
def set_contains(
    item: Set[Any],
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether list 'item' contains the type(s) in 'contents'.

    Args:
        item (Set[Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return serial_contains(item = item, contents = contents)

@contains.register(tuple)   
def tuple_contains(
    item: tuple[Any, ...],
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether tuple 'item' contains the type(s) in 'contents'.

    Args:
        item (tuple[Any, ...]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    if isinstance(contents, tuple) and len(item) == len(contents):
        technique = parallel_contains
    else:
        technique = serial_contains
    return technique(item = item, contents = contents)

@contains.register(Sequence)   
def parallel_contains(
    item: Sequence[Any],
    contents: tuple[Type[Any], ...]) -> bool:
    """Returns whether parallel 'item' contains the type(s) in 'contents'.

    Args:
        item (Sequence[Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return all(isinstance(item[i], contents[i]) for i in enumerate(item))

@contains.register(Container)       
def serial_contains(
    item: Container[Any],
    contents: Union[Type[Any], tuple[Type[Any], ...]]) -> bool:
    """Returns whether serial 'item' contains the type(s) in 'contents'.

    Args:
        item (Collection[Any]): item to examine.
        contents (Union[Type[Any], tuple[Type[Any], ...]]): types to check for
            in 'item' contents.

    Returns:
        bool: whether 'item' holds the types in 'contents'.
        
    """
    return all(isinstance(i, contents) for i in item)
         
def get_annotations(
    item: object, 
    include_private: bool = False) -> dict[str, Type[Any]]:
    """Returns dict of attributes of 'item' with type annotations.
    
    Args:
        item (object): instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        dict[str, Any]: dict of attributes in 'item' (keys are attribute names 
            and values are type annotations) that are type annotated.
            
    """
    annotations = item.__annotations__
    if include_private:
        return annotations
    else:
        return {k: v for k, v in annotations.items() if not k.startswith('_')}

def get_attributes(
    item: object, 
    include_private: bool = False) -> dict[str, Any]:
    """Returns dict of attributes of 'item'.
    
    Args:
        item (Any): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        dict[str, Any]: dict of attributes in 'item' (keys are attribute names 
            and values are attribute values).
            
    """
    attributes = name_attributes(item = item, include_private = include_private)
    values = [getattr(item, m) for m in attributes]
    return dict(zip(attributes, values))

def get_methods(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> dict[str, types.MethodType]:
    """Returns dict of methods of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.

    Returns:
        dict[str, types.MethodType]: dict of methods in 'item' (keys are method 
            names and values are methods).
        
    """ 
    methods = name_methods(item = item, include_private = include_private)
    return [getattr(item, m) for m in methods]

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
                return utilities.snakify(item.__name__) # type: ignore
            except AttributeError:
                if item.__class__.__name__ is not None:
                    return utilities.snakify( # type: ignore
                        item.__class__.__name__) 
                else:
                    return default

def get_properties(
    item: object, 
    include_private: bool = False) -> dict[str, Any]:
    """Returns properties of 'item'.

    Args:
        item (object): instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.

    Returns:
        dict[str, Any]: dict of properties in 'item' (keys are property names 
            and values are property values).
        
    """    
    properties = name_properties(item = item, include_private = include_private)
    values = [getattr(item, p) for p in properties]
    return dict(zip(properties, values))

def get_signatures(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> dict[str, inspect.Signature]:
    """Returns dict of method signatures of 'item'.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.

    Returns:
        dict[str, inspect.Signature]: dict of method signatures in 'item' (keys 
            are method names and values are method signatures).
                   
    """ 
    methods = name_methods(item = item, include_private = include_private)
    signatures = [inspect.signature(getattr(item, m)) for m in methods]
    return dict(zip(methods, signatures))

def get_variables(
    item: object, 
    include_private: bool = False) -> dict[str, Any]:
    """Returns dict of attributes of 'item' that are not methods or properties.
    
    Args:
        item (object): instance to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        dict[str, Any]: dict of attributes in 'item' (keys are attribute names 
            and values are attribute values) that are not methods or properties.
            
    """
    attributes = name_attributes(item = item, include_private = include_private)
    methods = name_methods(item = item, include_private = include_private)
    properties = name_properties(item = item, include_private = include_private)
    variables = [
        a for a in attributes if a not in methods and a not in properties]
    values = [getattr(item, m) for m in variables]
    return dict(zip(variables, values))

def has_attributes(
    item: Union[object, Type[Any]], 
    attributes: MutableSequence[str]) -> bool:
    """Returns whether 'attributes' exist in 'item'.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        attributes (MutableSequence[str]): names of attributes to check to see
            if they exist in 'item'.
            
    Returns:
        bool: whether all 'attributes' exist in 'items'.
    
    """
    return all(hasattr(item, a) for a in attributes)

def has_methods(
    item: Union[object, Type[Any]], 
    methods: Union[str, MutableSequence[str]]) -> bool:
    """Returns whether 'item' has 'methods' which are methods.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        methods (Union[str, MutableSequence[str]]): name(s) of methods to check 
            to see if they exist in 'item' and are types.MethodType.
            
    Returns:
        bool: whether all 'methods' exist in 'items' and are types.MethodType.
        
    """
    methods = list(utilities.iterify(methods))
    return all(is_method(item = item, attribute = m) for m in methods)

def has_properties(
    item: Union[object, Type[Any]], 
    properties: Union[str, MutableSequence[str]]) -> bool:
    """Returns whether 'item' has 'properties' which are properties.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        properties (MutableSequence[str]): names of properties to check to see 
            if they exist in 'item' and are property type.
            
    Returns:
        bool: whether all 'properties' exist in 'items'.
        
    """
    properties = list(utilities.iterify(properties))
    return all(is_property(item = item, attribute = p) for p in properties)

def has_signatures(
    item: Union[object, Type[Any]], 
    signatures: Mapping[str, inspect.Signature]) -> bool:
    """Returns whether 'item' has 'signatures' of its methods.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        signatures (Mapping[str, inspect.Signature]): keys are the names of 
            methods and values are the corresponding method signatures.
            
    Returns:
        bool: whether all 'signatures' exist in 'items'.
        
    """
    item_signatures = get_signatures(item = item, include_private = True)
    pass_test = True
    for name, parameters in signatures.items():
        if (name not in item_signatures or item_signatures[name] != parameters):
            pass_test = False
    return pass_test
    
def has_traits(
    item: Union[object, Type[Any]],
    attributes: Optional[MutableSequence[str]] = None,
    methods: Optional[MutableSequence[str]] = None,
    properties: Optional[MutableSequence[str]] = None,
    signatures: Optional[Mapping[str, inspect.Signature]] = None) -> bool:
    """Returns if 'item' has 'attributes', 'methods' and 'properties'.

    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        attributes (MutableSequence[str]): names of attributes to check to see
            if they exist in 'item'.
        methods (MutableSequence[str]): name(s) of methods to check to see if 
            they exist in 'item' and are types.MethodType.          
        properties (MutableSequence[str]): names of properties to check to see 
            if they exist in 'item' and are property type.
        signatures (Mapping[str, inspect.Signature]): keys are the names of 
            methods and values are the corresponding method signatures.
                          
    Returns:
        bool: whether all passed arguments exist in 'items'.    
    
    """
    if not inspect.isclass(item):
        item = item.__class__ 
    attributes = attributes or []
    methods = methods or []
    properties = properties or []
    signatures = signatures or {}
    return (
        has_attributes(item = item, attributes = attributes)
        and has_methods(item = item, methods = methods)
        and has_properties(item = item, properties = properties)
        and has_signatures(item = item, signatures = signatures))
    
@functools.singledispatch
def has_types(item: object) -> Optional[Union[
    tuple[Type[Any], ...], 
    tuple[tuple[Type[Any], ...], tuple[Type[Any], ...]]]]:
    """Returns types contained in 'item'.

    Args:
        item (object): item to examine.
    
    Returns:
        Optional[Union[tuple[Type[Any], ...], tuple[tuple[Type[Any], ...], 
            tuple[Type[Any], ...]]]]:: returns the types of things contained 
            in 'item'. Returns None if 'item' is not a container.
        
    """
    raise TypeError(f'item {item} is not supported by {__name__}')

@has_types.register(Mapping)  
def has_types_dict(
    item: Mapping[Hashable, Any]) -> Optional[
        tuple[tuple[Type[Any], ...], tuple[Type[Any], ...]]]:
    """Returns types contained in 'item'.

    Args:
        item (object): item to examine.
    
    Returns:
        Optional[tuple[Type[Any], ...]]: returns the types of things contained 
            in 'item'. Returns None if 'item' is not a container.
        
    """
    if isinstance(item, Mapping):
        key_types = has_types_sequence(item = item.keys())
        value_types = has_types_sequence(item = item.values())
        return tuple([key_types, value_types])
    else:
        return None

@has_types.register(MutableSequence)  
def has_types_list(item: list[Any]) -> Optional[tuple[Type[Any], ...]]:
    """Returns types contained in 'item'.

    Args:
        item (list[Any]): item to examine.
    
    Returns:
        Optional[tuple[Type[Any], ...]]: returns the types of things contained 
            in 'item'. Returns None if 'item' is not a container.
        
    """
    if isinstance(item, list):
        key_types = has_types_sequence(item = item.keys())
        value_types = has_types_sequence(item = item.values())
        return tuple([key_types, value_types])
    else:
        return None

@has_types.register(Sequence)    
def has_types_sequence(item: Sequence[Any]) -> Optional[tuple[Type[Any], ...]]:
    """Returns types contained in 'item'.

    Args:
        item (Sequence[Any]): item to examine.
    
    Returns:
        Optional[tuple[Type[Any], ...]]: returns the types of things contained 
            in 'item'. Returns None if 'item' is not a container.
        
    """
    if isinstance(item, Sequence):
        all_types = []
        for thing in item:
            kind = type(thing)
            if not kind in all_types:
                all_types.append(kind)
        return tuple(all_types)
    else:
        return None
 
def is_class_attribute(item: Union[object, Type[Any]], attribute: str) -> bool:
    """Returns if 'attribute' is a class attribute of 'item'."""
    if not inspect.isclass(item):
        item = item.__class__
    return (
        hasattr(item, attribute)
        and not is_method(item = item, attribute = attribute)
        and not is_property(item = item, attribute = attribute))
    
def is_container(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is a container and not a str.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a container but not a str.
        
    """  
    if not inspect.isclass(item):
        item = item.__class__ 
    return issubclass(item, Container) and not issubclass(item, str)

def is_function(item: Union[object, Type[Any]], attribute: Any) -> bool:
    """Returns if 'attribute' is a function of 'item'."""
    if isinstance(attribute, str):
        try:
            attribute = getattr(item, attribute)
        except AttributeError:
            return False
    return isinstance(attribute, types.FunctionType)
   
def is_iterable(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is iterable and is NOT a str type.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is iterable but not a str.
        
    """ 
    if not inspect.isclass(item):
        item = item.__class__ 
    return issubclass(item, Iterable) and not issubclass(item, str)
        
def is_method(item: Union[object, Type[Any]], attribute: Any) -> bool:
    """Returns if 'attribute' is a method of 'item'."""
    if isinstance(attribute, str):
        try:
            attribute = getattr(item, attribute)
        except AttributeError:
            return False
    return inspect.ismethod(attribute)

def is_nested(item: Mapping[Any, Any]) -> bool:
    """Returns if 'item' is nested at least one-level.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a nested mapping.
        
    """ 
    return (
        isinstance(item, Mapping) 
        and any(isinstance(v, Mapping) for v in item.values()))
 
def is_property(item: Union[object, Type[Any]], attribute: Any) -> bool:
    """Returns if 'attribute' is a property of 'item'."""
    if not inspect.isclass(item):
        item = item.__class__
    if isinstance(attribute, str):
        try:
            attribute = getattr(item, attribute)
        except AttributeError:
            return False
    return isinstance(attribute, property)

def is_sequence(item: Union[object, Type[Any]]) -> bool:
    """Returns if 'item' is a sequence and is NOT a str type.
    
    Args:
        item (Union[object, Type[Any]]): class or instance to examine.
        
    Returns:
        bool: if 'item' is a sequence but not a str.
        
    """ 
    if not inspect.isclass(item):
        item = item.__class__ 
    return issubclass(item, Sequence) and not issubclass(item, str) 

def is_variable(item: Union[object, Type[Any]], attribute: str) -> bool:
    """Returns if 'attribute' is a simple data attribute of 'item'.

    Args:
        item (Union[object, Type[Any]]): [description]
        attribute (str): [description]

    Returns:
        bool: [description]
        
    """
    return (
        hasattr(item, attribute)
        and not is_function(item = item, attribute = attribute)
        and not is_property(item = item, attribute = attribute))

def name_attributes(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> list[str]:
    """Returns attribute names of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        list[str]: names of attributes in 'item'.
            
    """
    names = dir(item)
    if not include_private:
        names = utilities.drop_privates(item = names)
    return names

def name_methods(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> list[str]:
    """Returns method names of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        list[str]: names of methods in 'item'.
            
    """
    methods = [
        a for a in dir(item)
        if is_method(item = item, attribute = a)]
    if not include_private:
        methods = utilities.drop_privates(item = methods)
    return methods

def name_parameters(item: Type[Any]) -> list[str]:
    """Returns list of parameters based on annotations of 'item'.

    Args:
        item (Type[Any]): class to get parameters to.

    Returns:
        list[str]: names of parameters in 'item'.
        
    """          
    return list(item.__annotations__.keys())

def name_properties(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> list[str]:
    """Returns method names of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        list[str]: names of properties in 'item'.
            
    """
    if not inspect.isclass(item):
        item = item.__class__
    properties = [
        a for a in dir(item)
        if is_property(item = item, attribute = a)]
    if not include_private:
        properties = utilities.drop_privates(item = properties)
    return properties

def name_variables(
    item: Union[object, Type[Any]], 
    include_private: bool = False) -> list[str]:
    """Returns variable names of 'item'.
    
    Args:
        item (Union[object, Type[Any]]): item to examine.
        include_private (bool): whether to include items that begin with '_'
            (True) or to exclude them (False). Defauls to False.
                        
    Returns:
        list[str]: names of attributes in 'item' that are neither methods nor
            properties.
            
    """
    names = [a for a in dir(item) if is_variable(item = item, attribute = a)]
    if not include_private:
        names = utilities.drop_privates(item = names)
    return names



# def has_annotations(
#     item: Union[object, Type[Any]], 
#     attributes: Mapping[str, Type[Any]]) -> bool:
#     """Returns whether 'attributes' exist in 'item' and are the right type.
    
#     Args:
#         item (Union[object, Type[Any]]): class or instance to examine.
#         attributes (dict[str, Type[Any]]): dict where keys are the attribute 
#             names and values are the expected types of whose named attributes.
            
#     Returns
#         bool: whether all of the 'attributes' exist in 'item' and are of the
#             proper type.
            
#     """
#     matched = True
#     if inspect.isclass(item):
#         for attribute, value in attributes.items():
#             if value is not None:
#                 try:
#                     testing = getattr(item, attribute)
#                     testing = item.__annotations__[testing]
#                 except AttributeError:
#                     return False
#                 try:
#                     if not issubclass(testing, value):
#                         return False
#                 except TypeError:
#                     pass
#     else:
#         for attribute, value in attributes.items():
#             if value is not None:
#                 try:
#                     testing = getattr(item, attribute)
#                 except AttributeError:
#                     return False
#                 try:
#                     if not isinstance(testing, value):
#                         return False
#                 except TypeError:
#                     pass
#     return matched  
