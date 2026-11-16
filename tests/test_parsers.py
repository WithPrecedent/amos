"""Tests the parsers module."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

import amos
from amos import parsers

NEWLINE = chr(10)


@pytest.fixture
def opinions() -> amos.Dataset:
    """A small dataset of court opinions, as `load_files` makes them."""
    texts = [
        NEWLINE.join([
            'United States v. Smith',
            'PRIOR HISTORY: Appeal from the District of Maine.',
            "JUDGES: Before LYNCH, O'SCANNLAIN, and Barron, Circuit Judges.",
            'OPINION: We REVERSE. See 512 F.3d 1093 and 98 F.4th 12.']),
        NEWLINE.join([
            'Jones v. Acme Corp.',
            'JUDGES: BOUDIN',
            'OPINION: Affirmed. See 5 F.3d 2.']),
        'In re Doe',
        None]
    data = pd.DataFrame({
        'file': ['a.txt', 'b.txt', 'c.txt', 'd.txt'],
        'text': texts,
        'court': ['First Circuit', 'ninth circuit', 'Ninth Circuit', None]})
    return amos.Dataset(data.astype('str'))


@pytest.fixture(autouse = True)
def restore_rulebooks() -> Iterator[None]:
    """Removes the rulebooks that a test registers."""
    saved = dict(parsers._RULEBOOKS)
    yield
    parsers._RULEBOOKS.clear()
    parsers._RULEBOOKS.update(saved)


def _parse(dataset: amos.Dataset, *rules: dict) -> pd.DataFrame:
    """Applies `rules` to `dataset` and returns its data."""
    parsers.ParseDocuments().apply(dataset, rules = list(rules))
    return dataset.data


def test_flag_count_and_map_rules(opinions: amos.Dataset) -> None:
    data = _parse(
        opinions,
        {'kind': 'flag', 'name': 'reversed', 'pattern': 'revers',
         'flags': 'ignorecase'},
        {'kind': 'flag', 'name': 'reversed', 'pattern': 'vacat'},
        {'kind': 'count', 'name': 'cites', 'pattern': r'F\.\dd'},
        {'kind': 'count', 'name': 'cites', 'pattern': r'F\.4th'},
        {'kind': 'map', 'column': 'court', 'name': 'circuit',
         'pattern': 'first', 'value': '1', 'flags': 'ignorecase'},
        {'kind': 'map', 'column': 'court', 'name': 'circuit',
         'pattern': 'Ninth', 'value': 9},
        {'kind': 'map', 'column': 'court', 'name': 'circuit',
         'pattern': 'circuit', 'value': 0, 'flags': 'ignorecase'})
    assert data['reversed'].tolist() == [True, False, False, False]
    assert data['cites'].tolist() == [2, 1, 0, 0]
    assert data['circuit'].tolist()[:3] == [1, 0, 9]
    assert pd.isna(data['circuit'].iloc[3])
    assert opinions.history[-1]['created'] == ['reversed', 'cites', 'circuit']
    assert opinions.history[-1]['changed'] == []


def test_map_all_rules_join_their_values_once(opinions: amos.Dataset) -> None:
    data = _parse(
        opinions,
        {'kind': 'map_all', 'name': 'outcome', 'pattern': 'REVERSE',
         'value': 'reversed'},
        {'kind': 'map_all', 'name': 'outcome', 'pattern': 'affirm',
         'value': 'affirmed', 'flags': 'ignorecase'},
        {'kind': 'map_all', 'name': 'outcome', 'pattern': 'See',
         'value': 'cites'},
        {'kind': 'map_all', 'name': 'outcome', 'pattern': 'We REVERSE',
         'value': 'reversed'})
    assert data['outcome'].tolist()[:2] == [
        'reversed, cites', 'affirmed, cites']
    assert data['outcome'].isna().tolist()[2:] == [True, True]
    assert pd.api.types.is_string_dtype(data['outcome'].dtype)


def test_extract_rules(opinions: amos.Dataset) -> None:
    cite = r'\d+ F\.\w+ \d+'
    data = _parse(
        opinions,
        {'kind': 'extract', 'name': 'first_cite', 'pattern': cite},
        {'kind': 'extract', 'name': 'first_cite', 'pattern': 'In re \\w+'},
        {'kind': 'extract_all', 'name': 'cites', 'pattern': cite},
        {'kind': 'extract_all', 'name': 'cites', 'pattern': r'(?i)in (re) \w+'},
        {'kind': 'extract', 'name': 'verb', 'pattern': r'(?i)we (\w+)'})
    assert data['first_cite'].tolist()[:3] == [
        '512 F.3d 1093', '5 F.3d 2', 'In re Doe']
    assert data['cites'].tolist()[:3] == [
        '512 F.3d 1093, 98 F.4th 12', '5 F.3d 2', 'In re Doe']
    # The whole match is kept, even if the pattern has groups.
    assert data['verb'].tolist()[0] == 'We REVERSE'
    assert data['verb'].isna().tolist() == [False, True, True, True]


def test_split_fields_and_names_rules(opinions: amos.Dataset) -> None:
    data = _parse(
        opinions,
        {'kind': 'split', 'name': 'party1, party2', 'pattern': r' v\. '},
        {'kind': 'split', 'column': 'party2', 'name': 'caption, rest',
         'pattern': NEWLINE},
        {'kind': 'fields', 'pattern': '^[A-Z][A-Z ]+:', 'flags': 'multiline'},
        {'kind': 'fields', 'name': 'header', 'pattern': '^[A-Z][A-Z ]+:',
         'flags': 'multiline'},
        {'kind': 'names', 'column': 'judges', 'name': 'panel',
         'pattern': r'BEFORE|CIRCUIT JUDGES|,|\bAND\b'})
    assert data['party1'].tolist()[:3] == ['United States', 'Jones', 'In re Doe']
    assert data['party2'].isna().tolist()[2:] == [True, True]
    assert data['caption'].tolist()[:2] == ['Smith', 'Acme Corp.']
    assert data['prior_history'].tolist()[:2] == [
        'Appeal from the District of Maine.', np.nan]
    assert data['header_prior_history'].tolist()[0] == (
        'Appeal from the District of Maine.')
    assert data['opinion'].tolist()[:2] == [
        'We REVERSE. See 512 F.3d 1093 and 98 F.4th 12.',
        'Affirmed. See 5 F.3d 2.']
    assert data['panel'].tolist()[:2] == ['LYNCH, OSCANNLAIN, BARRON', 'BOUDIN']
    assert data['panel'].isna().tolist()[2:] == [True, True]


def test_replace_rules(opinions: amos.Dataset) -> None:
    data = _parse(
        opinions,
        {'kind': 'replace', 'name': 'cited', 'pattern': r'(\d+) F\.(\w+) \d+',
         'value': r'[\1 \2]'},
        {'kind': 'replace', 'column': 'court', 'pattern': r'\s*circuit',
         'flags': 'ignorecase'},
        {'kind': 'flag', 'column': 'court', 'name': 'ninth',
         'pattern': '^ninth$', 'flags': 'ignorecase'})
    assert data['cited'].tolist()[1].endswith('See [5 3d].')
    assert data['court'].tolist()[:3] == ['First', 'ninth', 'Ninth']
    assert data['ninth'].tolist() == [False, True, True, False]
    assert opinions.history[-1]['changed'] == ['court']
    assert opinions.history[-1]['created'] == ['cited', 'ninth']


def test_rules_search_the_parser_column_and_made_columns(
    opinions: amos.Dataset) -> None:
    data = opinions.data.rename(columns = {'text': 'body'})
    result = parsers.ParseDocuments().apply(
        data,
        column = 'body',
        rules = [
            {'kind': 'extract', 'name': 'judges', 'pattern': 'JUDGES:.*'},
            {'kind': 'flag', 'column': 'judges', 'name': 'boudin',
             'pattern': 'BOUDIN'}])
    assert result.data['boudin'].tolist() == [False, True, False, False]
    with pytest.raises(KeyError, match = 'flag rule for x.*not in the data'):
        parsers.ParseDocuments().apply(
            result, rules = [{'kind': 'flag', 'name': 'x', 'pattern': 'a'}])
    with pytest.raises(TypeError, match = 'is not text'):
        parsers.ParseDocuments().apply(
            result, rules = [
                {'kind': 'flag', 'column': 'boudin', 'name': 'x',
                 'pattern': 'a'}])


def test_parsers_keep_rows_with_repeated_labels(
    opinions: amos.Dataset) -> None:
    opinions.data.index = [0, 0, 1, 1]
    data = _parse(
        opinions,
        {'kind': 'extract_all', 'name': 'cites', 'pattern': r'F\.\w+'},
        {'kind': 'fields', 'pattern': '^[A-Z]+:', 'flags': 'multiline'},
        {'kind': 'split', 'name': 'p1, p2', 'pattern': r' v\. '})
    assert list(data.index) == [0, 0, 1, 1]
    assert data['cites'].tolist()[:2] == ['F.3d, F.4th', 'F.3d']
    assert data['judges'].tolist()[1] == 'BOUDIN'
    assert data['p1'].tolist()[:2] == ['United States', 'Jones']


def test_verbose_patterns_can_end_in_a_comment(
    opinions: amos.Dataset) -> None:
    data = _parse(
        opinions,
        {'kind': 'extract', 'name': 'cite', 'flags': 'verbose',
         'pattern': r'\d+ \s F\.3d \s \d+  # a citation to F.3d'})
    assert data['cite'].tolist()[:2] == ['512 F.3d 1093', '5 F.3d 2']


def test_rules_from_csv_files(
    opinions: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    folder = tmp_path / 'rules'
    folder.mkdir()
    (folder / 'b_outcomes.csv').write_text(
        'column,name,kind,pattern,value,flags,note\n'
        ',reversed,flag,revers,,ignorecase,Reversals\n'
        'court,circuit,map,first,1,ignorecase,\n',
        encoding = 'utf-8-sig')
    (folder / 'a_parties.csv').write_text(
        'kind,name,pattern\nsplit,"party1, party2",\\sv\\.\\s\n',
        encoding = 'utf-8')
    book = parsers.Rulebook.create(folder, folder / 'a_parties.csv')
    assert [r.kind for r in book] == ['split', 'flag', 'map']
    assert book.name == 'a_parties, b_outcomes'
    assert book.names == ['party1', 'party2', 'reversed', 'circuit']
    assert book.columns == ['court']
    assert book.rules[1].note == 'Reversals'
    parsers.ParseDocuments().apply(opinions, rules = str(folder))
    data = opinions.data
    assert data['party1'].tolist()[0] == 'United States'
    assert data['reversed'].tolist() == [True, False, False, False]
    assert data['circuit'].tolist()[0] == 1


def test_rules_in_courtpy_layout(
    opinions: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'header.csv'
    path.write_text(
        'target,variable,kind,pattern,value,ignorecase,dotall,note\n'
        'text,"party1, party2",split,\\Wv\\.\\W,,FALSE,TRUE,\n'
        'text,opinion,section,OPINION:.*,,FALSE,TRUE,\n'
        'opinion,decision,label,affirm,affirmed,TRUE,FALSE,\n'
        'opinion,decision,label,revers,reversed,TRUE,FALSE,\n'
        'opinion,cites,matches,\\d+ F\\.\\w+ \\d+,,FALSE,FALSE,\n'
        'opinion,agencies,labels,See,citation,FALSE,FALSE,\n'
        'opinion,,remove,See,,FALSE,FALSE,\n',
        encoding = 'utf-8')
    book = parsers.Rulebook.create(path)
    assert [r.kind for r in book] == [
        'split', 'extract', 'map', 'map', 'extract_all', 'map_all',
        'replace']
    assert book.rules[0].pattern.flags & re.DOTALL
    parsers.ParseDocuments().apply(opinions, rules = book)
    data = opinions.data
    assert data['decision'].tolist()[:3] == ['reversed', 'affirmed', np.nan]
    assert data['cites'].tolist()[0] == '512 F.3d 1093, 98 F.4th 12'
    assert data['agencies'].tolist()[:2] == ['citation', 'citation']
    # A "remove" rule replaces its matches with a space, as courtpy's did.
    assert data['opinion'].tolist()[1] == 'OPINION: Affirmed.   5 F.3d 2.'


def test_registered_rulebooks(
    opinions: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'flags.csv'
    path.write_text(
        'kind,name,pattern\nflag,reversed,REVERSE\n', encoding = 'utf-8')
    amos.register_rulebook('flags', path)
    amos.register_rulebook('all', [
        'flags', {'kind': 'count', 'name': 'words', 'pattern': r'\S+'}])
    book = parsers.Rulebook.create('all')
    assert len(book) == 2
    assert book.name == 'all'
    settings = {
        'general': {'seed': 43},
        'study_project': {'study_workers': 'wrangler'},
        'wrangler': {'techniques': 'parse_documents'},
        'parse_documents_parameters': {'rules': 'all'}}
    project = amos.Project.create(settings, item = opinions.data)
    result = project.result.data
    assert result['reversed'].tolist() == [True, False, False, False]
    assert result['words'].tolist()[2:] == [3, 0]
    assert amos.library.classify('parse_documents') == 'parser'
    amos.register_rulebook('loop', ['loop'])
    with pytest.raises(ValueError, match = 'names itself'):
        parsers.Rulebook.create('loop')
    with pytest.raises(FileNotFoundError, match = 'registered rulebook'):
        parsers.Rulebook.create('missing')
    with pytest.raises(ValueError, match = 'needs a name'):
        amos.register_rulebook(' ', path)


@pytest.mark.parametrize(('row', 'message'), [
    ({'kind': 'guess', 'name': 'x', 'pattern': 'a'}, 'is not one of'),
    ({'kind': 'flag', 'name': 'x', 'pattern': ''}, 'pattern is blank'),
    ({'kind': 'flag', 'name': 'x', 'pattern': '('}, 'is not valid'),
    ({'kind': 'flag', 'name': '', 'pattern': 'a'}, 'name is blank'),
    ({'kind': 'flag', 'name': 'x, y', 'pattern': 'a'}, 'makes one column'),
    ({'kind': 'split', 'name': 'x', 'pattern': 'a'}, 'needs two names'),
    ({'kind': 'flag', 'name': 'x', 'pattern': 'a', 'value': 1},
     'does not use a value'),
    ({'kind': 'map', 'name': 'x', 'pattern': 'a'}, 'needs a value'),
    ({'kind': 'map_all', 'name': 'x', 'pattern': 'a', 'value': 'b, c'},
     'has a comma'),
    ({'kind': 'flag', 'name': 'x', 'pattern': 'a', 'flags': 'loud'},
     'is not one of'),
    ({'kind': 'fields', 'pattern': '(?P<label>A):'}, 'named groups'),
    ({'kind': 'replace', 'column': 'a, b', 'name': 'x', 'pattern': 'a'},
     'changes one column')])
def test_invalid_rules(row: dict, message: str) -> None:
    with pytest.raises(ValueError, match = message):
        parsers.Rulebook.create([row])


def test_invalid_rule_files(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'rules.csv'
    path.write_text(
        'kind,name,pattern,value\n\nflag,x,a,\nmap,y,b,\n', encoding = 'utf-8')
    with pytest.raises(ValueError, match = r'rules\.csv, row 4: .*value'):
        parsers.Rulebook.create(path)
    path.write_text('name,pattern\nx,a\n', encoding = 'utf-8')
    with pytest.raises(ValueError, match = r"missing the columns \['kind'\]"):
        parsers.Rulebook.create(path)
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match = 'no CSV files'):
        parsers.Rulebook.create(empty)


def test_parse_documents_without_rules_changes_nothing(
    opinions: amos.Dataset) -> None:
    before = opinions.data.copy()
    parsers.ParseDocuments().apply(opinions)
    parsers.ParseDocuments().apply(opinions, rules = [])
    pd.testing.assert_frame_equal(opinions.data, before)


def test_rulebooks_combine(tmp_path: pathlib.Path) -> None:
    first = parsers.Rulebook.create(
        [{'kind': 'flag', 'name': 'x', 'pattern': 'a'}])
    first.name = 'first'
    second = parsers.Rulebook(name = 'second')
    combined = first + second
    assert len(combined) == 1
    assert combined.name == 'first, second'
    assert list(combined) == first.rules
    path = tmp_path / 'rules.csv'
    path.write_text('kind,name,pattern\nflag,x,a\n', encoding = 'utf-8')
    # A file named twice is read once.
    assert len(parsers.Rulebook.create(path, [path], str(path))) == 1
