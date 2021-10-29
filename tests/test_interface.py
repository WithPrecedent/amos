"""
test_project: tests Project class and created composite objects
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import annotations
import dataclasses
import pathlib
from typing import (Any, Callable, ClassVar, Dict, Iterable, list, Mapping, 
                    Optional, Sequence, tuple, Type, Union)

import denovo

import amos


@dataclasses.dataclass
class Parser(amos.workers.Contest):

    pass


@dataclasses.dataclass
class Search(amos.workers.Step):

    pass   


@dataclasses.dataclass
class Divide(amos.workers.Step):

    pass   
    
    
@dataclasses.dataclass
class Destroy(amos.workers.Step):

    pass   
    

@dataclasses.dataclass
class Slice(amos.workers.Technique):

    pass  


@dataclasses.dataclass
class Dice(amos.workers.Technique):

    pass 
    
    
@dataclasses.dataclass
class Find(amos.workers.Technique):

    pass 

    
@dataclasses.dataclass
class Locate(amos.workers.Technique):

    pass 

    
@dataclasses.dataclass
class Explode(amos.workers.Technique):

    pass 

    
@dataclasses.dataclass
class Dynamite(amos.workers.Technique):
    
    name: str = 'annihilate'


def test_project():
    project = amos.Project.create(
        name = 'cool_project',
        settings = pathlib.Path('tests') / 'project_settings.py',
        automatic = True)
    # Tests base libraries.
    assert 'parser' in amos.workers.Component.library.subclasses
    dynamite = Dynamite()
    assert 'annihilate' in amos.workers.Component.library.instances
    # Tests workflow construction.
    print('test project workflow', project.workflow)
    print('test workflow endpoints', str(project.workflow.endpoints))
    print('test workflow roots', str(project.workflow.roots))
    return


if __name__ == '__main__':
    denovo.testing.testify(target_module = amos.interface, 
                           testing_module = __name__)
    