"""
convert: functions that convert types
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


ToDo:

    
"""
from __future__ import annotations
import functools
from typing import Any, Callable, Optional, Type, Union

from . import convert


def bondafide(
    _wrapped: Optional[Type[Any]] = None, 
    *,
    include: Optional[list[str]] = None, 
    exclude: Optional[list[str]] = None) -> Any:
    """Wraps a python dataclass and validates/converts attributes.
    
    """
    include = include or []
    exclude = exclude or []
    def validator(wrapped: Type[Any]) -> Any:
        @functools.wraps(wrapped)
        def wrapper(*args: Any, **kwargs: Any) -> object:
            kwargs.update(convert.kwargify(args = args, item = wrapped))
            instance = wrapped(**kwargs)
            attributes = include or wrapped.__annotations__.keys()
            attributes = [a for a in attributes if a not in exclude] # type: ignore
            for attribute in attributes:
                try:
                    kind = wrapped.__annotations__[attribute]
                    key = kind.__name__
                    value = getattr(instance, attribute)
                    if not isinstance(value, kind):
                        converter = convert.catalog[key]
                        new_value = converter(item = value)
                        setattr(instance, attribute, new_value)
                except KeyError:
                    pass
            return instance
        return wrapper
    if _wrapped is None:
        return validator
    else:
        return validator(wrapped = _wrapped)
    
   