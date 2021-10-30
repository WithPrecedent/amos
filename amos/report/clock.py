"""
clock: date and time related tools
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
    how_soon_is_now (Callable): converts a current date and time to a str.
    timer (Callable): computes the time it takes for the wrapped 'process' to
        complete.  
        
ToDo:
    Provide mechanisms for 'timer' to record results in logger and/or the python
        terminal.

"""
from __future__ import annotations
from collections.abc import Callable
import datetime
import time
from typing import Any, Optional, Type, Union


""" General Tools """

def datetime_string(
    prefix: str = '', 
    time_format: str = '%Y-%m-%d_%H-%M') -> str:
    return ''.join([prefix, datetime.datetime.now().strftime(time_format)])

def how_soon_is_now(
    prefix: Optional[str] = None,
    time_format: str = '%Y-%m-%d_%H-%M') -> str:
    """Creates a string from current date and time.

    Args:
        prefix: a prefix to add to the returned str.
        
    Returns:
        str: with current date and time in 'format' format.

    """
    time_string = datetime.datetime.now().strftime(time_format)
    if prefix is None:
        return f'{prefix}{time_string}'
    else:
        return time_string

""" Decorators """

def timer(process: Callable[..., Optional[Any]]) -> (
    Callable[..., Optional[Any]]):
    """Decorator for computing the length of time a process takes.

    Args:
        process (Callable[..., Optional[Any]]): wrapped callable to compute the 
            time it takes to complete its execution.

    """
    try:
        name = process.__name__
    except AttributeError:
        name = process.__class__.__name__
    def shell_timer(operation: Callable[..., Optional[Any]]) -> (
        Callable[..., Optional[Any]]):
        def decorated(*args: Any, **kwargs: Any) -> (
            Callable[..., Optional[Any]]):
            def convert_time(
                seconds: Union[int, float]) -> tuple[int, int, int]:
                minutes, seconds = divmod(seconds, 60)
                hours, minutes = divmod(minutes, 60)
                return int(hours), int(minutes), int(seconds)
            implement_time = time.time()
            result = operation(*args, **kwargs)
            total_time = time.time() - implement_time
            h, m, s = convert_time(total_time)
            print(f'{name} completed in %d:%02d:%02d' % (h, m, s))
            return result
        return decorated
    return shell_timer
