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

import fiat


@dataclasses.dataclass
class Parser(fiat.workers.Contest):

    pass


@dataclasses.dataclass
class Search(fiat.workers.Step):

    pass   


@dataclasses.dataclass
class Divide(fiat.workers.Step):

    pass   
    
    
@dataclasses.dataclass
class Destroy(fiat.workers.Step):

    pass   
    

@dataclasses.dataclass
class Slice(fiat.workers.Technique):

    pass  


@dataclasses.dataclass
class Dice(fiat.workers.Technique):

    pass 
    
    
@dataclasses.dataclass
class Find(fiat.workers.Technique):

    pass 

    
@dataclasses.dataclass
class Locate(fiat.workers.Technique):

    pass 

    
@dataclasses.dataclass
class Explode(fiat.workers.Technique):

    pass 

    
@dataclasses.dataclass
class Dynamite(fiat.workers.Technique):
    
    name: str = 'annihilate'


def test_project():
    project = fiat.Project.create(
        name = 'cool_project',
        settings = pathlib.Path('tests') / 'project_settings.py',
        automatic = True)
    # Tests base libraries.
    assert 'parser' in fiat.workers.Component.library.subclasses
    dynamite = Dynamite()
    assert 'annihilate' in fiat.workers.Component.library.instances
    # Tests workflow construction.
    print('test project workflow', project.workflow)
    print('test workflow endpoints', str(project.workflow.endpoints))
    print('test workflow roots', str(project.workflow.roots))
    return


if __name__ == '__main__':
    denovo.testing.testify(target_module = fiat.interface, 
                           testing_module = __name__)
    