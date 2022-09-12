"""
hybrid: lightweight linear composite data structures
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
    
          
To Do:
    Add methods needed for Composite subclassing.
    
"""
from __future__ import annotations
import abc
from collections.abc import (
    Collection, Hashable, Mapping, MutableMapping, MutableSequence, Sequence)
import dataclasses
from typing import Any, Callable, ClassVar, Optional, Type, TypeVar, Union

from . import mappings
from . import sequences
from ..observe import report
from ..change import modify
from . import composites
from ..observe import check
 
