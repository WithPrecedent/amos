"""Tests the parsers module."""

from __future__ import annotations

import pathlib
import dataclasses
import re
import warnings
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import amos
from amos import parsers
from amos.wrangler.parsers import _base as parsers_base
from amos.wrangler.parsers import _rulebooks

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
def restore_library() -> Iterator[None]:
    """Removes the parsers (and so rulebooks and kinds) that a test defines."""
    saved = set(amos.library.all)
    yield
    for name in set(amos.library.all) - saved:
        amos.library.delete(name)


def _parse(dataset: amos.Dataset, *rules: dict) -> pd.DataFrame:
    """Applies `rules` to `dataset` and returns its data."""
    parsers.ParseAll().apply(dataset, rules = list(rules))
    return dataset.data


def test_flag_count_and_map_rules(opinions: amos.Dataset) -> None:
    data = _parse(
        opinions,
        {'kind': 'flags', 'name': 'reversed', 'pattern': 'revers',
         'flags': 'ignorecase'},
        {'kind': 'flags', 'name': 'reversed', 'pattern': 'vacat'},
        {'kind': 'counts', 'name': 'cites', 'pattern': r'F\.\dd'},
        {'kind': 'counts', 'name': 'cites', 'pattern': r'F\.4th'},
        {'kind': 'category', 'column': 'court', 'name': 'circuit',
         'pattern': 'first', 'value': '1', 'flags': 'ignorecase'},
        {'kind': 'category', 'column': 'court', 'name': 'circuit',
         'pattern': 'Ninth', 'value': 9},
        {'kind': 'category', 'column': 'court', 'name': 'circuit',
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
        {'kind': 'categories', 'name': 'outcome', 'pattern': 'REVERSE',
         'value': 'reversed'},
        {'kind': 'categories', 'name': 'outcome', 'pattern': 'affirm',
         'value': 'affirmed', 'flags': 'ignorecase'},
        {'kind': 'categories', 'name': 'outcome', 'pattern': 'See',
         'value': 'cites'},
        {'kind': 'categories', 'name': 'outcome', 'pattern': 'We REVERSE',
         'value': 'reversed'})
    assert data['outcome'].tolist()[:2] == [
        'reversed, cites', 'affirmed, cites']
    assert data['outcome'].isna().tolist()[2:] == [True, True]
    assert pd.api.types.is_string_dtype(data['outcome'].dtype)


def test_extract_rules(opinions: amos.Dataset) -> None:
    cite = r'\d+ F\.\w+ \d+'
    data = _parse(
        opinions,
        {'kind': 'match', 'name': 'first_cite', 'pattern': cite},
        {'kind': 'match', 'name': 'first_cite', 'pattern': 'In re \\w+'},
        {'kind': 'matches', 'name': 'cites', 'pattern': cite},
        {'kind': 'matches', 'name': 'cites', 'pattern': r'(?i)in (re) \w+'},
        {'kind': 'match', 'name': 'verb', 'pattern': r'(?i)we (\w+)'})
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
        {'kind': 'flags', 'column': 'court', 'name': 'ninth',
         'pattern': '^ninth$', 'flags': 'ignorecase'})
    assert data['cited'].tolist()[1].endswith('See [5 3d].')
    assert data['court'].tolist()[:3] == ['First', 'ninth', 'Ninth']
    assert data['ninth'].tolist() == [False, True, True, False]
    assert opinions.history[-1]['changed'] == ['court']
    assert opinions.history[-1]['created'] == ['cited', 'ninth']


def test_rules_search_the_parser_column_and_made_columns(
    opinions: amos.Dataset) -> None:
    data = opinions.data.rename(columns = {'text': 'body'})
    result = parsers.ParseAll().apply(
        data,
        column = 'body',
        rules = [
            {'kind': 'match', 'name': 'judges', 'pattern': 'JUDGES:.*'},
            {'kind': 'flags', 'column': 'judges', 'name': 'boudin',
             'pattern': 'BOUDIN'}])
    assert result.data['boudin'].tolist() == [False, True, False, False]
    with pytest.raises(KeyError, match = 'flags rule for x.*not in the data'):
        parsers.ParseAll().apply(
            result, rules = [{'kind': 'flags', 'name': 'x', 'pattern': 'a'}])
    with pytest.raises(TypeError, match = 'is not text'):
        parsers.ParseAll().apply(
            result, rules = [
                {'kind': 'flags', 'column': 'boudin', 'name': 'x',
                 'pattern': 'a'}])


def test_parsers_keep_rows_with_repeated_labels(
    opinions: amos.Dataset) -> None:
    opinions.data.index = [0, 0, 1, 1]
    data = _parse(
        opinions,
        {'kind': 'matches', 'name': 'cites', 'pattern': r'F\.\w+'},
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
        {'kind': 'match', 'name': 'cite', 'flags': 'verbose',
         'pattern': r'\d+ \s F\.3d \s \d+  # a citation to F.3d'})
    assert data['cite'].tolist()[:2] == ['512 F.3d 1093', '5 F.3d 2']


def test_rules_from_csv_files(
    opinions: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    folder = tmp_path / 'rules'
    folder.mkdir()
    (folder / 'b_outcomes.csv').write_text(
        'column,name,kind,pattern,value,flags,note\n'
        ',reversed,flags,revers,,ignorecase,Reversals\n'
        'court,circuit,category,first,1,ignorecase,\n',
        encoding = 'utf-8-sig')
    (folder / 'a_parties.csv').write_text(
        'kind,name,pattern\nsplit,"party1, party2",\\sv\\.\\s\n',
        encoding = 'utf-8')
    book = parsers.Rulebook.create(folder, folder / 'a_parties.csv')
    assert [r.kind for r in book] == ['split', 'flags', 'category']
    assert book.name == 'a_parties, b_outcomes'
    assert book.names == ['party1', 'party2', 'reversed', 'circuit']
    assert book.columns == ['court']
    assert book.rules[1].note == 'Reversals'
    parsers.ParseAll().apply(opinions, rules = str(folder))
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
        'split', 'match', 'category', 'category', 'matches', 'categories',
        'replace']
    assert book.rules[0].pattern.flags & re.DOTALL
    parsers.ParseAll().apply(opinions, rules = book)
    data = opinions.data
    assert data['decision'].tolist()[:3] == ['reversed', 'affirmed', np.nan]
    assert data['cites'].tolist()[0] == '512 F.3d 1093, 98 F.4th 12'
    assert data['agencies'].tolist()[:2] == ['citation', 'citation']
    # A "remove" rule replaces its matches with a space, as courtpy's did.
    assert data['opinion'].tolist()[1] == 'OPINION: Affirmed.   5 F.3d 2.'


def test_subclasses_of_parse_all_are_rulebooks(
    opinions: amos.Dataset,
    tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'flags.csv'
    path.write_text(
        'kind,name,pattern\nflags,reversed,REVERSE\n', encoding = 'utf-8')

    @dataclasses.dataclass
    class ParseOutcomes(parsers.ParseAll):
        rules: ClassVar[Any] = path

    @dataclasses.dataclass
    class ParseEverything(parsers.ParseAll):
        rules: ClassVar[Any] = [
            'outcomes', {'kind': 'counts', 'name': 'words', 'pattern': r'\S+'}]

    for source in ('everything', 'parse_everything', ParseEverything):
        book = parsers.Rulebook.create(source)
        assert len(book) == 2
        assert book.name == 'everything'
    assert amos.library.classify('parse_everything') == 'parser'
    for techniques, parameters in [
        ('parse_all', {'rules': 'everything'}), ('parse_everything', {})]:
        settings = {
            'general': {'seed': 43},
            'study_project': {'study_workers': 'wrangler'},
            'wrangler': {'techniques': techniques},
            f'{techniques}_parameters': parameters}
        project = amos.Project.create(settings, item = opinions.data.copy())
        result = project.result.data
        assert result['reversed'].tolist() == [True, False, False, False]
        assert result['words'].tolist()[2:] == [3, 0]

    @dataclasses.dataclass
    class ParseLoop(parsers.ParseAll):
        rules: ClassVar[Any] = ['loop']

    with pytest.raises(ValueError, match = 'names itself'):
        parsers.Rulebook.create('loop')
    with pytest.raises(FileNotFoundError, match = 'or a rulebook'):
        parsers.Rulebook.create('missing')
    with pytest.raises(ValueError, match = 'has no rules'):
        parsers.Rulebook.create(parsers.ParseAll)


@pytest.mark.parametrize(('row', 'message'), [
    ({'kind': 'guess', 'name': 'x', 'pattern': 'a'}, 'is not one of'),
    ({'kind': 'flags', 'name': 'x', 'pattern': ''}, 'pattern is blank'),
    ({'kind': 'flags', 'name': 'x', 'pattern': '('}, 'is not valid'),
    ({'kind': 'flags', 'name': '', 'pattern': 'a'}, 'name is blank'),
    ({'kind': 'flags', 'name': 'x, y', 'pattern': 'a'}, 'makes one column'),
    ({'kind': 'split', 'name': 'x', 'pattern': 'a'}, 'two or more names'),
    ({'kind': 'flags', 'name': 'x', 'pattern': 'a', 'value': 1},
     'does not use a value'),
    ({'kind': 'category', 'name': 'x', 'pattern': 'a'}, 'needs a value'),
    ({'kind': 'categories', 'name': 'x', 'pattern': 'a', 'value': 'b, c'},
     'which separates the values'),
    ({'kind': 'flags', 'name': 'x', 'pattern': 'a', 'flags': 'loud'},
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
        'kind,name,pattern,value\n\nflags,x,a,\ncategory,y,b,\n',
        encoding = 'utf-8')
    with pytest.raises(ValueError, match = r'rules\.csv, row 4: .*value'):
        parsers.Rulebook.create(path)
    path.write_text('name,pattern\nx,a\n', encoding = 'utf-8')
    with pytest.raises(ValueError, match = r"missing the columns \['kind'\]"):
        parsers.Rulebook.create(path)
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match = 'no CSV files'):
        parsers.Rulebook.create(empty)


def test_parse_all_without_rules_changes_nothing(
    opinions: amos.Dataset) -> None:
    before = opinions.data.copy()
    parsers.ParseAll().apply(opinions)
    parsers.ParseAll().apply(opinions, rules = [])
    pd.testing.assert_frame_equal(opinions.data, before)


def test_rulebooks_combine(tmp_path: pathlib.Path) -> None:
    first = parsers.Rulebook.create(
        [{'kind': 'flags', 'name': 'x', 'pattern': 'a'}])
    first.name = 'first'
    second = parsers.Rulebook(name = 'second')
    combined = first + second
    assert len(combined) == 1
    assert combined.name == 'first, second'
    assert list(combined) == first.rules
    path = tmp_path / 'rules.csv'
    path.write_text('kind,name,pattern\nflags,x,a\n', encoding = 'utf-8')
    # A file named twice is read once.
    assert len(parsers.Rulebook.create(path, [path], str(path))) == 1


def test_old_kinds_are_read_as_the_kinds_they_became() -> None:
    old = ['flag', 'count', 'map', 'map_all', 'extract', 'extract_all']
    rows = [
        {'kind': kind, 'name': 'x', 'pattern': 'a',
         'value': 'v' if kind.startswith('map') else None}
        for kind in old]
    book = parsers.Rulebook.create(rows)
    assert [r.kind for r in book] == [
        'flags', 'counts', 'category', 'categories', 'match', 'matches']
    assert [r.parser for r in book] == [
        parsers.ParseFlags, parsers.ParseCounts, parsers.ParseCategory,
        parsers.ParseCategories, parsers.ParseMatch, parsers.ParseMatches]


def test_pattern_parsers_work_without_rules(opinions: amos.Dataset) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        parsers.ParseFlags().apply(
            opinions, name = 'reversed', pattern = ['(revers)', 'vacat'],
            flags = 'ignorecase')
    parsers.ParseCounts().apply(
        opinions, name = 'cites', pattern = [r'F\.\dd', r'F\.4th'])
    parsers.ParseCategory().apply(
        opinions, column = 'court', name = 'circuit', pattern = 'ninth',
        value = '9', flags = re.IGNORECASE)
    parsers.ParseCategories().apply(
        opinions, name = 'outcome', pattern = ['REVERSE', 'REVERSE'],
        value = 'reversed')
    parsers.ParseMatch().apply(
        opinions, name = 'first_cite', pattern = r'\d+ F\.\w+ \d+')
    parsers.ParseMatches().apply(
        opinions, name = 'cites_found', pattern = r'\d+ F\.\w+ \d+')
    parsers.ParseFields().apply(
        opinions, pattern = '^[A-Z][A-Z ]+:', flags = ['multiline'])
    parsers.ParseNames().apply(
        opinions, column = 'judges', name = 'panel',
        pattern = r'BEFORE|CIRCUIT JUDGES|,|\bAND\b')
    parsers.ParseReplace().apply(
        opinions, column = 'court', pattern = r'\s*circuit',
        flags = 'ignorecase')
    data = opinions.data
    assert data['reversed'].tolist() == [True, False, False, False]
    assert data['cites'].tolist() == [2, 1, 0, 0]
    assert data['circuit'].tolist()[1:3] == [9, 9]
    assert data['outcome'].tolist()[0] == 'reversed'
    assert data['first_cite'].tolist()[:2] == ['512 F.3d 1093', '5 F.3d 2']
    assert data['cites_found'].tolist()[0] == '512 F.3d 1093, 98 F.4th 12'
    assert data['panel'].tolist()[:2] == ['LYNCH, OSCANNLAIN, BARRON', 'BOUDIN']
    assert data['court'].tolist()[:3] == ['First', 'ninth', 'Ninth']
    assert opinions.history[0]['technique'] == 'parse_flags'
    assert amos.library.classify('parse_matches') == 'pattern_parser'
    assert amos.library.classify('parse_all') == 'parser'


def test_pattern_parsers_search_several_and_categorical_columns(
    opinions: amos.Dataset) -> None:
    opinions.data['court'] = opinions.data['court'].astype('category')
    parsers.ParseFlags().apply(
        opinions, column = 'text, court', name = 'ninth', pattern = 'Ninth')
    assert opinions.data['ninth'].tolist() == [False, False, True, False]


def test_parse_split_makes_any_number_of_pieces(
    opinions: amos.Dataset) -> None:
    parsers.ParseSplit().apply(
        opinions, name = 'caption, rest, last', pattern = NEWLINE)
    data = opinions.data
    assert data['caption'].tolist()[:3] == [
        'United States v. Smith', 'Jones v. Acme Corp.', 'In re Doe']
    assert data['rest'].tolist()[1] == 'JUDGES: BOUDIN'
    assert data['last'].tolist()[1] == 'OPINION: Affirmed. See 5 F.3d 2.'
    assert data['last'].tolist()[0].startswith('JUDGES:')
    assert data['rest'].isna().tolist() == [False, False, True, True]


def test_pattern_parsers_check_their_parameters(
    opinions: amos.Dataset) -> None:
    before = opinions.data.copy()
    for kind in parsers_base._rule_kinds().values():
        kind().apply(opinions)
    parsers.ParseAll().apply(opinions)
    pd.testing.assert_frame_equal(opinions.data, before)
    with pytest.raises(ValueError, match = 'name is blank'):
        parsers.ParseFlags().apply(opinions, pattern = 'a')
    with pytest.raises(ValueError, match = 'needs a value'):
        parsers.ParseCategory().apply(opinions, name = 'x', pattern = 'a')
    with pytest.raises(ValueError, match = 'two or more names'):
        parsers.ParseSplit().apply(opinions, name = 'x', pattern = 'a')
    with pytest.raises(KeyError, match = 'flags rule for x.*not in the data'):
        parsers.ParseFlags().apply(
            opinions, column = 'missing', name = 'x', pattern = 'a')


def test_pattern_parsers_are_kinds_of_rules(
    opinions: amos.Dataset,
    tmp_path: pathlib.Path) -> None:

    @dataclasses.dataclass
    class ParseInitials(parsers.PatternParser):
        """Makes the first letter of each match, joined (for the tests)."""

        rule_aliases: ClassVar[tuple[str, ...]] = ('Map',)

        def find(
            self,
            rule: parsers.Rule,
            texts: Mapping[str, pd.Series],
            made: dict[str, pd.Series],
            index: pd.Index) -> None:
            """Makes the initials of the matches in each row."""
            text = next(iter(texts.values()))
            found = text.str.findall(rule.pattern.pattern, flags = rule.flags)
            made[rule.names[0]] = found.map(
                lambda m: ''.join(w[0] for w in m) if isinstance(m, list)
                else None).astype('str')

    @dataclasses.dataclass
    class ParseFirstLetters(ParseInitials):
        """Inherits the search of `ParseInitials`, but not its aliases."""

    assert ParseInitials.kind() == 'initials'
    kinds = parsers_base._rule_kinds()
    assert kinds['initials'] is ParseInitials
    assert kinds['first_letters'] is ParseFirstLetters
    # An alias takes the place of an old kind ("map" was "category").
    assert kinds['map'] is ParseInitials
    path = tmp_path / 'rules.csv'
    path.write_text(
        'kind,name,pattern\ninitials,initials,[A-Z]\\w+\nmap,caps,[A-Z]+\n',
        encoding = 'utf-8')
    parsers.ParseAll().apply(opinions, rules = path)
    assert opinions.data['initials'].tolist()[2] == 'ID'
    assert opinions.data['caps'].tolist()[2] == 'ID'
    assert ParseInitials().apply(
        opinions.data, column = 'court', name = 'c',
        pattern = '[A-Z]').data['c'].tolist()[:3] == ['FC', '', 'NC']
    amos.library.delete('parse_initials')
    with pytest.raises(ValueError, match = 'is not one of'):
        parsers.Rulebook.create([{'kind': 'initials', 'pattern': 'a'}])


def test_any_parser_can_be_written_without_rules(
    opinions: amos.Dataset) -> None:

    @dataclasses.dataclass
    class ParseLength(amos.Parser):

        def parse(self, data, **kwargs):
            data['length'] = data['text'].str.len()
            return data

    ParseLength().apply(opinions)
    assert opinions.data['length'].tolist()[2] == len('In re Doe')
    assert opinions.history[-1]['created'] == ['length']


def test_pattern_parsers_work_in_a_project(opinions: amos.Dataset) -> None:
    settings = {
        'general': {'seed': 43},
        'study_project': {'study_workers': 'wrangler'},
        'wrangler': {'techniques': 'parse_flags, parse_split'},
        'parse_flags_parameters': {
            'name': 'reversed', 'pattern': 'revers|vacat',
            'flags': 'ignorecase'},
        'parse_split_parameters': {
            'name': 'party1, party2', 'pattern': r'\sv\.\s'}}
    project = amos.Project.create(settings, item = opinions.data)
    result = project.result.data
    assert result['reversed'].tolist() == [True, False, False, False]
    assert result['party2'].tolist()[1].startswith('Acme Corp.')


def test_named_groups_make_a_column_each(opinions: amos.Dataset) -> None:
    cite = r'(?P<volume>\d+) F\.(?P<reporter>\w+) (?P<page>\d+)'
    parsers.ParseMatch().apply(opinions, pattern = cite)
    parsers.ParseMatches().apply(opinions, name = 'all', pattern = cite)
    data = opinions.data
    assert data['volume'].tolist()[:2] == ['512', '5']
    assert data['reporter'].tolist()[:2] == ['3d', '3d']
    assert data['page'].isna().tolist() == [False, False, True, True]
    assert data['all_volume'].tolist()[0] == '512, 98'
    assert data['all_reporter'].tolist()[0] == '3d, 4th'
    # Unnamed groups still keep the whole match.
    book = parsers.Rulebook.create([
        {'kind': 'match', 'name': 'verb', 'pattern': r'(?i)we (\w+)'},
        {'kind': 'match', 'pattern': r'(?P<year>\d{4})'}])
    assert book.rules[1].names == ()
    with pytest.raises(ValueError, match = 'needs none'):
        parsers.ParseMatch().apply(opinions, pattern = r'(\d+)')


def test_named_groups_come_from_one_column_per_row() -> None:
    data = pd.DataFrame({
        'a': ['x 1', None, 'nothing'],
        'b': ['y 2', 'y 3', 'z 4']}).astype('str')
    result = parsers.ParseMatch().apply(
        data, column = 'a, b',
        pattern = r'(?P<letter>[a-z]) (?P<digit>\d)').data
    assert result['letter'].tolist() == ['x', 'y', 'z']
    assert result['digit'].tolist() == ['1', '3', '4']


def test_separators(opinions: amos.Dataset) -> None:
    parsers.ParseMatches().apply(
        opinions, name = 'cites', pattern = r'\d+ F\.\w+ \d+',
        separator = '; ')
    parsers.ParseMatches().apply(
        opinions, column = 'court', name = 'letters', pattern = r'[A-Z]',
        separator = '')
    parsers.ParseCategories().apply(
        opinions, name = 'outcome', pattern = {'REVERSE': 'a,b', 'See': 'c'},
        separator = '|')
    data = opinions.data
    assert data['cites'].tolist()[0] == '512 F.3d 1093; 98 F.4th 12'
    assert data['letters'].tolist()[::2] == ['FC', 'NC']
    assert data['letters'].isna().tolist() == [False, True, False, True]
    assert data['outcome'].tolist()[:2] == ['a,b|c', 'c']
    judges = r'BEFORE|CIRCUIT JUDGES|,|\bAND\b'
    rules = [
        {'kind': 'fields', 'pattern': '^[A-Z][A-Z ]+:', 'flags': 'multiline'},
        {'kind': 'names', 'column': 'judges', 'name': 'panel',
         'pattern': judges, 'separator': '/'},
        {'kind': 'names', 'column': 'court', 'name': 'panel',
         'pattern': 'CIRCUIT', 'separator': '/'},
        {'kind': 'names', 'column': 'judges', 'name': 'panel',
         'pattern': judges, 'separator': '/'}]
    parsers.ParseAll().apply(opinions, rules = rules)
    assert opinions.data['panel'].tolist()[:3] == [
        'LYNCH/OSCANNLAIN/BARRON/FIRST', 'BOUDIN/NINTH', 'NINTH']
    with pytest.raises(ValueError, match = "has '\\|'"):
        parsers.ParseCategories().apply(
            opinions, name = 'x', pattern = 'a', value = 'b|c',
            separator = '|')


def test_patterns_mapped_to_values_and_defaults(
    opinions: amos.Dataset) -> None:
    parsers.ParseCategory().apply(
        opinions, column = 'court', name = 'circuit',
        pattern = {'first': 1, 'ninth': 9}, flags = 'ignorecase')
    parsers.ParseCategory().apply(
        opinions, name = 'outcome', default = 'unknown',
        pattern = {'revers': 'reversed', 'affirm': 'affirmed'},
        flags = 'ignorecase')
    parsers.ParseReplace().apply(
        opinions, column = '*', pattern = {r'\bv\.': 'versus', 'In re': None})
    data = opinions.data
    assert data['circuit'].tolist()[:3] == [1, 9, 9]
    assert data['outcome'].tolist() == [
        'reversed', 'affirmed', 'unknown', 'unknown']
    assert data['text'].tolist()[1].startswith('Jones versus Acme')
    assert data['text'].tolist()[2] == ' Doe'
    assert data['file'].tolist()[0] == 'a.txt'
    with pytest.raises(ValueError, match = 'given twice'):
        parsers.ParseCategory().apply(
            opinions, name = 'x', pattern = {'a': 1}, value = 2)


def test_parsers_can_search_every_column_of_text(
    opinions: amos.Dataset) -> None:
    parsers.ParseFlags().apply(
        opinions, column = None, name = 'ninth', pattern = '(?i)ninth')
    parsers.ParseCounts().apply(
        opinions, column = '*', name = 'txt', pattern = r'\.txt')
    assert opinions.data['ninth'].tolist() == [False, True, True, False]
    assert opinions.data['txt'].tolist() == [1, 1, 1, 1]
    empty = amos.Dataset(pd.DataFrame({'n': [1, 2]}))
    parsers.ParseFlags().apply(
        empty, column = '*', name = 'x', pattern = 'a')
    assert list(empty.data.columns) == ['n']
