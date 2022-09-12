"""
convert: functions that convert types
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

All tools should follow one of two forms. For conversion of a known type
to another type, the function name should be:
    f'{item type}_to_{output type}'
For a conversion from an unknown type to another type, the function name should
be:
    f'to_{output type}'
     
Contents:
    instancify: converts a class to an instance or adds kwargs to a passed 
        instance as attributes.
    listify: converts passed item to a list.
    namify: returns hashable name for passed item.
    numify: attempts to convert passed item to a numerical type.
    pathlibify: converts a str to a pathlib object or leaves it as
        a pathlib object.
    tuplify: converts a passed item to a tuple.
    typify: converts a str type to other common types, if possible.
    to_dict:
    to_index
    str_to_index
    to_int
    str_to_int
    float_to_int
    to_list
    str_to_list
    to_float
    int_to_float
    str_to_float
    to_path
    str_to_path
    to_str
    int_to_str
    float_to_str
    list_to_str
    none_to_str
    path_to_str
    datetime_to_str
    to_adjacency
    edges_to_adjacency
    matrix_to_adjacency
    pipeline_to_adjacency
    pipelines_to_adjacency
    tree_to_adjacency
    nodes_to_adjacency
    to_edges
    adjacency_to_edges
    to_matrix
    adjacency_to_matrix
    to_tree
    matrix_to_tree  
    
ToDo:
    Add more flexible tools.
    
"""
from __future__ import annotations
import ast
import collections
from collections.abc import (
    Collection, Hashable, Iterable, Mapping, MutableMapping, MutableSequence, 
    Sequence, Set)
import datetime
import inspect
import pathlib
from typing import Any, Callable, Optional, Type, TYPE_CHECKING, Union

import more_itertools

from . import modify
from ..observe import check

if TYPE_CHECKING:
    from ..containers import composites
    from ..containers import graph
    from ..containers import hybrid
    from ..containers import sequences
    from ..containers import tree
    

""" General Converters """

def instancify(item: Union[Type[Any], object], **kwargs: Any) -> Any:
    """Returns 'item' as an instance with 'kwargs' as parameters/attributes.
    
    If 'item' is already an instance, kwargs are added as attributes to the
    existing 'item'. This will overwrite any existing attributes of the same
    name.

    Args:
        item (Union[Type[Any], object])): class to make an instance out of by 
            passing kwargs or an instance to add kwargs to as attributes.

    Raises:
        TypeError: if 'item' is neither a class nor instance.
        
    Returns:
        object: a class instance with 'kwargs' as attributes or passed as 
            parameters (if 'item' is a class).
        
    """         
    if inspect.isclass(item):
        return item(**kwargs) # type: ignore
    elif isinstance(item, object):
        for key, value in kwargs.items():
            setattr(item, key, value)
        return item
    else:
        raise TypeError('item must be a class or class instance')
                  
def iterify(item: Any) -> Iterable:
    """Returns 'item' as an iterable, but does not iterate str types.
    
    Args:
        item (Any): item to turn into an iterable

    Returns:
        Iterable: of 'item'. A str type will be stored as a single item in an
            Iterable wrapper.
        
    """     
    if item is None:
        return iter(())
    elif isinstance(item, (str, bytes)):
        return iter([item])
    else:
        try:
            return iter(item)
        except TypeError:
            return iter((item,))
        
def kwargify(item: Type[Any], args: tuple[Any]) -> dict[Hashable, Any]:
    """Converts args to kwargs.
    
    Args:
        args (tuple): arguments without keywords passed to 'item'.
        item (Type): the item with annotations used to construct kwargs.
    
    Raises:
        ValueError: if there are more args than annotations in 'item'.
        
    Returns
        dict[Hashable, Any]: kwargs based on 'args' and 'item'.
    
    """
    annotations = list(item.__annotations__.keys())
    if len(args) > len(annotations):
        raise ValueError('There are too many args for item')
    else:
        return dict(zip(annotations, args))
    
def listify(item: Any, default: Optional[Any] = None) -> Any:
    """Returns passed item as a list (if not already a list).

    Args:
        item (Any): item to be transformed into a list to allow proper 
            iteration.
        default (Optional[Any]): the default value to return if 'item' is None.
            Unfortunately, to indicate you want None to be the default value,
            you need to put 'None' in quotes. If not passed, 'default' is set to 
            [].

    Returns:
        Any: a passed list, 'item' converted to a list, or the 'default' 
            argument.

    """
    if item is None:
        if default is None:
            return []
        elif default in ['None', 'none']:
            return None
        else:
            return default
    elif isinstance(item, MutableSequence) and not isinstance(item, str):
        return item
    else:
        return [item]

def namify(item: Any, default: Optional[str] = None) -> Optional[str]:
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
    elif (
        hasattr(item, 'name') 
        and not inspect.isclass(item)
        and isinstance(item.name, str)):
        return item.name
    else:
        try:
            return modify.snakify(item.__name__)
        except AttributeError:
            if item.__class__.__name__ is not None:
                return modify.snakify(item.__class__.__name__) 
            else:
                return default
                            
def numify(item: Any, raise_error: bool = False) -> Union[int, float, Any]:
    """Converts 'item' to a numeric type.
    
    If 'item' cannot be converted to a numeric type and 'raise_error' is False, 
        'item' is returned as is.

    Args:
        item (str): item to be converted.
        raise_error (bool): whether to raise a TypeError when conversion to a
            numeric type fails (True) or to simply return 'item' (False). 
            Defaults to False.

    Raises:
        TypeError: if 'item' cannot be converted to a numeric type and 
            'raise_error' is True.
            
    Returns
        Union[int, float, str]: converted to numeric type, if possible.

    """
    try:
        return int(item)
    except ValueError:
        try:
            return float(item)
        except ValueError:
            if raise_error:
                raise TypeError(
                    f'{item} not able to be converted to a numeric type')
            else:
                return item

def pathlibify(item: Union[str, pathlib.Path]) -> pathlib.Path:
    """Converts string 'path' to pathlib.Path object.

    Args:
        item (Union[str, pathlib.Path]): either a string summary of a
            path or a pathlib.Path object.

    Raises:
        TypeError if 'path' is neither a str or pathlib.Path type.

    Returns:
        pathlib.Path object.

    """
    if isinstance(item, str):
        return pathlib.Path(item)
    elif isinstance(item, pathlib.Path):
        return item
    else:
        raise TypeError('item must be str or pathlib.Path type')
           
def stringify(item: Any, default: Optional[Any] = None) -> Any:
    """Converts 'item' to a str from a Sequence.
    
    Args:
        item (Any): item to convert to a str from a list if it is a list.
        default (Any): value to return if 'item' is equivalent to a null
            value when passed. Defaults to None.
    
    Raises:
        TypeError: if 'item' is not a str or list-like object.
        
    Returns:
        Any: str, if item was a list, None or the default value if a null value
            was passed, or the item as it was passed if there previous two 
            conditions don't appply.

    """
    if item is None:
        if default is None:
            return ''
        elif default in ['None', 'none']: 
            return None
        else:
            return default
    elif isinstance(item, str):
        return item
    elif isinstance(item, Sequence):
        return ', '.join(item)
    else:
        raise TypeError('item must be str or a sequence')
    
def tuplify(item: Any, default: Optional[Any] = None) -> Any:
    """Returns passed item as a tuple (if not already a tuple).

    Args:
        item (Any): item to be transformed into a tuple.
        default (Any): the default value to return if 'item' is None.
            Unfortunately, to indicate you want None to be the default value,
            you need to put 'None' in quotes. If not passed, 'default'
            is set to ().

    Returns:
        tuple[Any]: a passed tuple, 'item' converted to a tuple, or 
            'default'.

    """
    if item is None:
        if default is None:
            return tuple()
        elif default in ['None', 'none']:
            return None
        else:
            return default
    elif isinstance(item, tuple):
        return item
    elif isinstance(item, Iterable):
        return tuple(item)
    else:
        return tuple([item])
        
def typify(item: str) -> Union[Sequence[Any], int, float, bool, str]:
    """Converts stings to appropriate, supported datatypes.

    The method converts strings to list (if ', ' is present), int, float,
    or bool datatypes based upon the content of the string. If no
    alternative datatype is found, the item is returned in its original
    form.

    Args:
        item (str): string to be converted to appropriate datatype.

    Returns:
        item (str, list, int, float, or bool): converted item.

    """
    if not isinstance(item, str):
        return item
    else:
        try:
            return int(item)
        except ValueError:
            try:
                return float(item)
            except ValueError:
                if item.lower() in ['true', 'yes']:
                    return True
                elif item.lower() in ['false', 'no']:
                    return False
                elif ', ' in item:
                    item = item.split(', ')
                    return [typify(i) for i in item]
                else:
                    return item

                         
""" Specific Converters """

# @amos.dynamic.dispatcher   
def to_dict(item: Any) -> MutableMapping[Hashable, Any]:
    """Converts 'item' to a MutableMapping.
    
    Args:
        item (Any): item to convert to a MutableMapping.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        MutableMapping: derived from 'item'.

    """
    if isinstance(item, MutableMapping):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')

# @amos.dynamic.dispatcher   
def to_index(item: Any) -> Hashable:
    """Converts 'item' to an Hashable.
    
    Args:
        item (Any): item to convert to a Hashable.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        Hashable: derived from 'item'.

    """
    if isinstance(item, Hashable):
        return item
    else:
        try:
            return hash(item)
        except TypeError:
            try:
                return str(item)
            except TypeError:
                try:
                    return modify.snakify(item.__name__)
                except AttributeError:
                    return modify.snakify(item.__class__.__name__)
                except AttributeError:
                    raise TypeError(f'item cannot be converted because it is ' 
                                    f'an unsupported type: '
                                    f'{type(item).__name__}')

# @to_index.register # type: ignore
def str_to_index(item: str) -> Hashable:
    """[summary]

    Args:
        item (str): [description]

    Returns:
        Hashable: [description]
    """    
    """Converts a str to an Hashable."""
    return item

# @amos.dynamic.dispatcher   
def to_int(item: Any) -> int:
    """Converts 'item' to a pathlib.Path.
    
    Args:
        item (Any): item to convert to a int.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        int: derived from 'item'.

    """
    if isinstance(item, int):
        return item
    else:
        raise TypeError(f'item cannot be converted because it is an '
                        f'unsupported type: {type(item).__name__}')

# @to_int.register # type: ignore
def str_to_int(item: str) -> int:
    """[summary]

    Args:
        item (str): [description]

    Returns:
        int: [description]
    """    
    """Converts a str to an int."""
    return int(item)

# @to_int.register # type: ignore
def float_to_int(item: float) -> int:
    """[summary]

    Args:
        item (float): [description]

    Returns:
        int: [description]
    """    
    """Converts a float to an int."""
    return int(item)

# @amos.dynamic.dispatcher   
def to_list(item: Any) -> list[Any]:
    """Converts 'item' to a list.
    
    Args:
        item (Any): item to convert to a list.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        list[Any]: derived from 'item'.

    """
    if isinstance(item, list[Any]):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')

# @to_list.register # type: ignore
def str_to_list(item: str) -> list[Any]:
    """[summary]

    Args:
        item (str): [description]

    Returns:
        list[Any]: [description]
    """    
    """Converts a str to a list."""
    return ast.literal_eval(item)

# @amos.dynamic.dispatcher   
def to_float(item: Any) -> float:
    """Converts 'item' to a float.
    
    Args:
        item (Any): item to convert to a float.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        float: derived from 'item'.

    """
    if isinstance(item, float):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')

# @to_float.register # type: ignore
def int_to_float(item: int) -> float:
    """[summary]

    Args:
        item (int): [description]

    Returns:
        float: [description]
    """    
    """Converts an int to a float."""
    return float(item)

# @to_float.register # type: ignore
def str_to_float(item: str) -> float:
    """[summary]

    Args:
        item (str): [description]

    Returns:
        float: [description]
    """    
    """Converts a str to a float."""
    return float(item)

# @amos.dynamic.dispatcher   
def to_path(item: Any) -> pathlib.Path:
    """Converts 'item' to a pathlib.Path.
    
    Args:
        item (Any): item to convert to a pathlib.Path.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        pathlib.Path: derived from 'item'.

    """
    if isinstance(item, pathlib.Path):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')

# @to_path.register # type: ignore   
def str_to_path(item: str) -> pathlib.Path:
    """[summary]

    Args:
        item (str): [description]

    Returns:
        pathlib.Path: [description]
    """    
    """Converts a str to a pathlib.Path."""
    return pathlib.pathlib.Path(item)

# @amos.dynamic.dispatcher   
def to_str(item: Any) -> str:
    """Converts 'item' to a str.
    
    Args:
        item (Any): item to convert to a str.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        str: derived from 'item'.

    """
    if isinstance(item, str):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')

# @to_str.register # type: ignore
def int_to_str(item: int) -> str:
    """[summary]

    Args:
        item (int): [description]

    Returns:
        str: [description]
    """    
    """Converts an int to a str."""
    return str(item)

# @to_str.register # type: ignore
def float_to_str(item: float) -> str:
    """[summary]

    Args:
        item (float): [description]

    Returns:
        str: [description]
    """    
    """Converts an float to a str."""
    return str(item)

# @to_str.register # type: ignore
def list_to_str(item: list[Any]) -> str:
    """[summary]

    Args:
        item (list[Any]): [description]

    Returns:
        str: [description]
    """    
    """Converts a list to a str."""
    return ', '.join(item)
   
# @to_str.register 
def none_to_str(item: None) -> str:
    """[summary]

    Args:
        item (None): [description]

    Returns:
        str: [description]
    """    
    """Converts None to a str."""
    return 'None'

# @to_str.register # type: ignore
def path_to_str(item: pathlib.Path) -> str:
    """[summary]

    Args:
        item (pathlib.Path): [description]

    Returns:
        str: [description]
    """    
    """Converts a pathlib.Path to a str."""
    return str(item)

# @to_str.register # type: ignore
def datetime_to_string(
    item: datetime.datetime,
    time_format: str = '%Y-%m-%d_%H-%M') -> str:
    return item.strftime(time_format)

""" Composite Converters """

# @amos.dispatcher 
def to_adjacency(item: Any) -> graph.Adjacency:
    """Converts 'item' to an Adjacency.
    
    Args:
        item (Any): item to convert to an Adjacency.

    Raises:
        TypeError: if 'item' is a type that is not registered with the 
        dispatcher.

    Returns:
        graph.Adjacency: derived from 'item'.

    """
    if check.is_adjacency(item = item):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')

# @to_adjacency.register # type: ignore
def edges_to_adjacency(item: graph.Edges) -> graph.Adjacency:
    """Converts 'item' to an Adjacency.

    Args:
        item (graph.Edges): item to convert to an Adjacency.

    Returns:
        graph.Adjacency: derived from 'item'.

    """
    adjacency = collections.defaultdict(set)
    for edge_pair in item:
        if edge_pair[0] not in adjacency:
            adjacency[edge_pair[0]] = {edge_pair[1]}
        else:
            adjacency[edge_pair[0]].add(edge_pair[1])
        if edge_pair[1] not in adjacency:
            adjacency[edge_pair[1]] = set()
    return adjacency

# @to_adjacency.register # type: ignore 
def matrix_to_adjacency(item: graph.Matrix) -> graph.Adjacency:
    """Converts 'item' to an Adjacency.

    Args:
        item (graph.Matrix): item to convert to an Adjacency.

    Returns:
        graph.Adjacency: derived from 'item'.

    """  
    matrix = item[0]
    names = item[1]
    name_mapping = dict(zip(range(len(matrix)), names))
    raw_adjacency = {
        i: [j for j, adjacent in enumerate(row) if adjacent] 
        for i, row in enumerate(matrix)}
    adjacency = collections.defaultdict(set)
    for key, value in raw_adjacency.items():
        new_key = name_mapping[key]
        new_values = set()
        for edge in value:
            new_values.add(name_mapping[edge])
        adjacency[new_key] = new_values
    return adjacency

# @to_adjacency.register # type: ignore 
def pipeline_to_adjacency(item: hybrid.Pipeline) -> graph.Adjacency:
    """Converts 'item' to an Adjacency.

    Args:
        item (hybrid.Pipeline): item to convert to an Adjacency.

    Returns:
        graph.Adjacency: derived from 'item'.

    """
    if not isinstance(item, (Collection)) or isinstance(item, str):
        item = [item]
    adjacency = collections.defaultdict(set)
    if len(item) == 1:
        adjacency.update({item: set()})
    else:
        edges = more_itertools.windowed(item, 2)
        for edge_pair in edges:
            if edge_pair[0] in adjacency:
                adjacency[edge_pair[0]].add(edge_pair[1])
            else:
                adjacency[edge_pair[0]] = {edge_pair[1]}
    return adjacency

# @to_adjacency.register # type: ignore 
def pipelines_to_adjacency(item: hybrid.Pipelines) -> graph.Adjacency:
    """Converts 'item' to an Adjacency.

    Args:
        item (hybrid.Pipelines): item to convert to an Adjacency.

    Returns:
        graph.Adjacency: derived from 'item'.

    """
    adjacency = collections.defaultdict(set)
    for _, pipeline in item.items():
        pipe_adjacency = pipeline_to_adjacency(item = pipeline)
        for key, value in pipe_adjacency.items():
            if key in adjacency:
                for inner_value in value:
                    if inner_value not in adjacency:
                        adjacency[key].add(inner_value)
            else:
                adjacency[key] = value
    return adjacency

# @to_adjacency.register # type: ignore 
def tree_to_adjacency(item: tree.Tree) -> graph.Adjacency:
    """Converts 'item' to an Adjacency.

    Args:
        item (tree.Tree): item to convert to an Adjacency.

    Returns:
        graph.Adjacency: derived from 'item'.

    """
    raise NotImplementedError
             
# @to_adjacency.register # type: ignore 
def nodes_to_adjacency(item: composites.Nodes) -> graph.Adjacency:
    """Converts 'item' to an Adjacency.

    Args:
        item (composites.Nodes): item to convert to an Adjacency.

    Returns:
        graph.Adjacency: derived from 'item'.

    """
    adjacency = collections.defaultdict(set)
    return adjacency.update((k, set()) for k in item)

# @amos.dispatcher   
def to_edges(item: Any) -> graph.Edges:
    """Converts 'item' to an Edges.
    
    Args:
        item (Any): item to convert to an Edges.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        graph.Edges: derived from 'item'.

    """
    if check.is_edges(item = item):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')
    
# @to_edges.register # type: ignore
def adjacency_to_edges(item: graph.Adjacency) -> graph.Edges:
    """Converts 'item' to an Edges.
    
    Args:
        item (graph.Adjacency): item to convert to an Edges.

    Returns:
        graph.Edges: derived from 'item'.

    """ 
    edges = []
    for node, connections in item.items():
        for connection in connections:
            edges.append(tuple([node, connection]))
    return tuple(edges)

# @amos.dispatcher   
def to_matrix(item: Any) -> graph.Matrix:
    """Converts 'item' to a Matrix.
    
    Args:
        item (Any): item to convert to a Matrix.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        graph.Matrix: derived from 'item'.

    """
    if check.is_matrix(item = item):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')

# @to_matrix.register # type: ignore 
def adjacency_to_matrix(item: graph.Adjacency) -> graph.Matrix:
    """Converts 'item' to a Matrix.
    
    Args:
        item (graph.Adjacency): item to convert to a Matrix.

    Returns:
        graph.Matrix: derived from 'item'.

    """ 
    names = list(item.keys())
    matrix = []
    for i in range(len(item)): 
        matrix.append([0] * len(item))
        for j in item[i]:
            matrix[i][j] = 1
    return tuple([matrix, names])    

# @amos.dispatcher   
def to_tree(item: Any) -> tree.Tree:
    """Converts 'item' to a Tree.
    
    Args:
        item (Any): item to convert to a Tree.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        tree.Tree: derived from 'item'.

    """
    if check.is_tree(item = item):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')

# @to_tree.register # type: ignore 
def matrix_to_tree(item: graph.Matrix) -> tree.Tree:
    """Converts 'item' to a Tree.
    
    Args:
        item (graph.Matrixy): item to convert to a Tree.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        tree.Tree: derived from 'item'.

    """
    tree = {}
    for node in item:
        children = item[:]
        children.remove(node)
        tree[node] = matrix_to_tree(children)
    return tree

# @amos.dispatcher   
def to_pipeline(item: Any) -> sequences.pipeline:
    """Converts 'item' to a Tree.
    
    Args:
        item (Any): item to convert to a Tree.

    Raises:
        TypeError: if 'item' is a type that is not registered.

    Returns:
        tree.Tree: derived from 'item'.

    """
    if check.is_tree(item = item):
        return item
    else:
        raise TypeError(
            f'item cannot be converted because it is an unsupported type: '
            f'{type(item).__name__}')