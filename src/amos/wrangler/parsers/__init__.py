r"""Techniques that turn the text in a dataset into data.

These are techniques of the "wrangler" stage of a project. A loader brings
documents into a dataset as they are (`load_files` makes a row for each file,
with its text in a "text" column), and parsers then find information in that
text and make columns of it, such as whether a contract can be renewed, the
city in an address, or the authors of a paper. Like a munger, a parser keeps
the rows and every column. Each search covers every row at once with the
vectorized methods of `pandas` (such as those of `Series.str`), so parsing is
fast even with many documents.

Each kind of search is a parser of its own, which a workflow can name in its
settings with a pattern and the columns to search and make (see
`PatternParser.parse`). Several values in one cell are separated by a comma
and a space unless the `separator` is set:

| Parser | Kind | Makes |
| --- | --- | --- |
| `parse_flags` | `flags` | True if the pattern appears, and otherwise false. |
| `parse_counts` | `counts` | The number of times the pattern appears. |
| `parse_category` | `category` | The `value` if the pattern appears. With several patterns (or rules), the first that matches wins, so they can turn text into categories (such as the name of a state into its abbreviation). |
| `parse_categories` | `categories` | The `value` of every pattern (or rule) that appears, once each, in order. A value cannot have the separator in it. |
| `parse_match` | `match` | The first match, or a column for each named group of the pattern (such as "(?P<year>\d{4})"), named for the group after the `name` if there is one. |
| `parse_matches` | `matches` | Every match, or a column for each named group as `parse_match` makes. |
| `parse_split` | `split` | A column for each name: the text before the first match, between the first and second, and so on, with the rest of the text in the last. Text without a match is all in the first. |
| `parse_names` | `names` | The names in the text, found by splitting the text wherever the pattern matches. The pattern should match the words and punctuation between names (such as "AUTHORS", "BY", "AND", and commas). The names are made upper-case without periods or apostrophes (so "O'Brien" is "OBRIEN"), line breaks always separate them, and pieces without two letters are left out. |
| `parse_fields` | `fields` | A column for each label that the pattern matches (such as "MEDICAL HISTORY:"), with the text after the label, up to the next label. The columns are named for the labels in snake case ("medical_history"), after the `name` if there is one. |
| `parse_replace` | `replace` | The text with every match replaced by the `value`, which can include the groups of the match ("\1" is the first). A blank `value` removes the matches. Without a `name`, the text it searches is changed. |

`parse_all` applies a rulebook: an ordered list of rules, each of which is
one of these searches, named by its kind. Rules are usually rows of CSV files
(spreadsheets), so they can be read and changed without knowing Python. They
can also be `dict`s in settings, with the same keys. The columns are:

| Column | Meaning |
| --- | --- |
| `column` | The columns of text to search, separated by commas. Defaults to the parser's "column" (such as "text") if blank. |
| `name` | The column that the rule makes (or adds to). A `split` rule names two or more, separated by commas. It is blank for a `replace` rule that changes the text it searches, and optional for a `fields` rule (whose columns are named for the labels in the text) and for a `match` or `matches` rule whose pattern has named groups. |
| `kind` | What the rule does: one of the kinds above, or the kind of a pattern parser of your own. |
| `pattern` | A regular expression to search for. |
| `value` | What a `category`, `categories`, or `replace` rule writes. Otherwise blank. |
| `flags` | Any of "ignorecase", "dotall" (so "." also matches line breaks), "multiline" (so "^" and "$" match at each line), and "verbose", separated by commas. Optional. |
| `separator` | What separates several values in one cell, for `categories`, `matches`, and `names` rules. Optional: a comma and a space if blank. |
| `note` | Anything you want to say about the rule. It is not used. Optional. |

Rules are applied from top to bottom, and every column that a rule makes can
be searched by the rules after it, so a rule can find a section of a
document (with `match`, `split`, or `fields`) for later rules to search.
Rules that make the same column are combined: flags are true if any rule
matches, counts are added, the first `category` or `match` that matches is
kept, and the results of `categories`, `matches`, and `names` rules are
joined. A text that is missing matches nothing. A parser given several
patterns combines them in the same way, as rules of the same name.

The kinds that this module first used (`flag`, `count`, `map`, `map_all`,
`extract`, and `extract_all`) are read as the kinds they became. CSV files in
an older layout of rules (with "target" for `column`, "variable" for `name`,
TRUE or FALSE in "ignorecase" and "dotall" columns, and the kinds "label",
"labels", "section", "matches", "excerpts", and "remove") are read too, as
the rules they stand for.

To write a parser, subclass `Parser` and write `parse`. To write a parser
that can also be a kind of rule, subclass `PatternParser` and write `find`:
the subclass is then a kind of rule, named for the class (`ParseInitials` is
"initials"). To name a rulebook, so that settings and other rulebooks can
use it, subclass `ParseAll` and set its `rules` (see `ParseAll`).

Contents:
    Parser: base class for techniques that turn text into data.
    PatternParser: base class for parsers that search text for a pattern, each
        of which can be a kind of rule.
    Rule: one search of text, as a row of a rulebook.
    ParseCategories: lists the value of every pattern that text matches, in
        order.
    ParseCategory: gives each text the value of the first pattern that it
        matches.
    ParseFields: makes a column for each labeled section of text.
    ParseNames: finds the names in text.
    ParseCounts: counts the matches of a pattern in text.
    ParseFlags: flags whether text matches a pattern.
    ParseMatch: keeps the first match of a pattern in text.
    ParseMatches: keeps every match of a pattern in text.
    ParseReplace: replaces the matches of a pattern in text.
    ParseSplit: splits text into columns where a pattern matches.
    ParseAll: finds information in text with a rulebook.
    Rulebook: an ordered collection of rules.

"""

from __future__ import annotations

import types

__all__: list[str] = [
    'ParseAll',
    'ParseCategories',
    'ParseCategory',
    'ParseCounts',
    'ParseFields',
    'ParseFlags',
    'ParseMatch',
    'ParseMatches',
    'ParseNames',
    'ParseReplace',
    'ParseSplit',
    'Parser',
    'PatternParser',
    'Rule',
    'Rulebook']

from ._base import Parser, PatternParser, Rule
from ._categories import ParseCategories, ParseCategory
from ._fields import ParseFields, ParseNames
from ._matches import (
    ParseCounts,
    ParseFlags,
    ParseMatch,
    ParseMatches,
    ParseReplace,
    ParseSplit,
)
from ._rulebooks import ParseAll, Rulebook

# The public names say that they are in `amos.parsers` (rather than in the
# private module that defines them, or in `amos.wrangler.parsers`), where they
# are imported from, so that records, pickles, and the documentation name
# them as `amos.parsers.Name`. `amos` makes `amos.parsers` importable.
for _name in __all__:
    if isinstance(globals()[_name], (type, types.FunctionType)):
        globals()[_name].__module__ = 'amos.parsers'
