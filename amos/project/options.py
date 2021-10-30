"""
options: base classes for a project
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
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

"""
from __future__ import annotations
import types
from typing import Any, Type

from . import filing
from . import nodes
from . import stages
from . import workshop


CLERK: Type[Any] = filing.Clerk
CONVERTERS: types.ModuleType = stages
DIRECTOR: Type[Any] = workshop.Director
LIBRARY: Type[Any] = nodes.ProjectLibrary
NODE: Type[Any] = nodes.Component
PARAMETERS: Type[Any] = nodes.Parameters
SETTINGS: Type[Any] = workshop.ProjectSettings
STAGE: Type[Any] = stages.Stage


def get_base(base_type: str) -> None:
    return globals()[base_type.upper()]

def set_base(base_type: str, base: Type[Any]) -> None:
    globals()[base_type.upper()] = base
    return