"""
test_amos: tests classes in amos
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

ToDo:
    
"""
from __future__ import annotations
import dataclasses

import amos


@dataclasses.dataclass
class TestClass(object):
    
    name: str = 'something'


@dataclasses.dataclass
class AnotherClass(object):
    
    name: str = 'another'


@dataclasses.dataclass
class ThirdClass(object):
    
    name: str = 'third'


# def test_proxy():
#     wrapped = TestClass()
#     proxy = amos.Proxy(contents = wrapped)
#     proxy.id = 4543
#     assert proxy.name == 'something'
#     assert proxy.id == 4543
#     assert hasattr(proxy, 'id')
#     del proxy.id
#     assert not hasattr(proxy, 'id')
#     proxy.name == 'something else'
#     print('test proxy', proxy.name)
#     assert proxy.name == 'something_else'
    

def test_listing():
    listing = amos.Listing(contents = ['a', 'b', 'c'])
    assert listing[1] == 'b'
    listing.add(item = 'd')
    assert listing[3] == 'd'
    listing.insert(2, 'zebra')
    assert listing[2] == 'zebra'
    listing.append('e')
    assert listing[5] == 'e'
    listing.extend(['f', 'g'])
    assert listing[7] == 'g'
    sub_listing = listing.subset(
        include = ['a', 'b', 'c', 'd', 'zebra'],
        exclude = 'd')
    assert sub_listing.contents == ['a', 'b', 'zebra', 'c']
    sub_listing.remove('c')
    assert sub_listing.contents == ['a', 'b', 'zebra']
    listing.clear()
    return

def test_hybrid():
    hybrid = amos.Hybrid(contents = ['a', 'b', 'c'])
    hybrid.setdefault(value = 'No')
    assert hybrid.get('tree') == 'No'
    assert hybrid[1] == 'b'
    hybrid.add(item = 'd')
    assert hybrid[3] == 'd'
    hybrid.insert(2, 'zebra')
    assert hybrid[2] == 'zebra'
    sub_hybrid = hybrid.subset(
        include = ['a', 'b', 'c', 'd', 'zebra'],
        exclude = 'd')
    assert sub_hybrid.contents == ['a', 'b', 'zebra', 'c']
    sub_hybrid.remove('c')
    assert sub_hybrid.contents == ['a', 'b', 'zebra']
    assert hybrid[0] == 'a'
    assert hybrid['zebra'] == 'zebra'
    hybrid.append('b')
    assert hybrid.values() == tuple(['a', 'b', 'zebra', 'c', 'd', 'b'])
    assert hybrid.keys() == tuple(['a', 'b', 'zebra', 'c', 'd', 'b'])
    hybrid.remove('b')
    assert hybrid.contents == ['a', 'zebra', 'c', 'd', 'b']
    hybrid.clear()
    test_class = TestClass()
    hybrid.add(test_class)
    assert hybrid.keys() == tuple(['something'])
    assert hybrid.values() == tuple([test_class])
    return

def test_dictionary():
    alt_created = amos.Dictionary.fromkeys(
        keys = ['a', 'b', 'c'], 
        value = 'tree')
    assert alt_created['a'] == 'tree'
    dictionary = amos.Dictionary(
        contents = {'a': 'b', 'c': 'd'}, 
        default_factory = 'Nada')
    assert dictionary.get('f') == 'Nada'
    assert dictionary['a'] == 'b'
    dictionary.add({'e': 'f'})
    assert dictionary['e'] == 'f'
    subset = dictionary.subset(include = ['a', 'e'])
    assert 'a' in subset.keys()
    assert 'b' not in subset.keys()
    assert 'a' not in subset.values()
    assert 'b' in subset.values()
    return

def test_catalog():
    catalog = amos.Catalog(contents = {'tester': TestClass})
    catalog.add({'another': AnotherClass})
    catalog.add({'a_third': ThirdClass()})
    assert 'tester' in catalog
    assert 'another' in catalog
    assert 'a_third' in catalog
    assert catalog['all'] == catalog['default']
    assert catalog['None'] == None
    catalog.default = 'tester'
    assert catalog['default'] == catalog['tester']
    del catalog[['tester', 'another']]
    assert 'tester' not in catalog
    assert len(catalog) == 1
    return
    
def test_library():
    library = amos.Library(classes = amos.Catalog(contents = {
        'tester': TestClass}))
    library.deposit(AnotherClass())
    library.deposit(ThirdClass, 'random_name')
    assert 'tester' in library
    assert issubclass(library['random_name'], ThirdClass)
    assert 'another_class' not in library.instances

    assert 'another_class' in library.classes
    assert 'another' in library.instances
    library.delete('random_name')
    # print('test library', library)
    assert 'random_name' not in library
    return
 
if __name__ == '__main__':
    # test_proxy()
    test_listing()
    test_hybrid()
    test_dictionary()
    test_catalog()
    test_library()
   