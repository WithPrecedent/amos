"""Tests the clerks module (Clerk and workspace)."""

from __future__ import annotations

import os
import pathlib

import pytest
from conftest import SEED, make_numeric

import amos
from amos import clerks, options


@pytest.fixture
def no_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removes the workspace variable that the tests set."""
    monkeypatch.delenv(options._WORKSPACE_VARIABLE, raising = False)


@pytest.mark.usefixtures('no_variable')
def test_the_workspace_holds_the_code_folder(tmp_path: pathlib.Path) -> None:
    project = tmp_path / 'code' / 'my_study' / 'src'
    project.mkdir(parents = True)
    assert clerks.workspace(project) == tmp_path
    assert clerks.workspace(tmp_path / 'code') == tmp_path


@pytest.mark.usefixtures('no_variable')
def test_without_a_code_folder_the_workspace_is_the_start(
    tmp_path: pathlib.Path) -> None:
    assert clerks.workspace(tmp_path) == tmp_path.resolve()
    os.chdir(tmp_path)
    assert clerks.workspace() == tmp_path.resolve()


def test_the_variable_and_the_option_name_the_workspace(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'code').mkdir()
    assert clerks.workspace(tmp_path / 'code') == tmp_path
    monkeypatch.setattr(options, '_DEFAULT_ROOT', tmp_path / 'chosen')
    assert clerks.workspace(tmp_path / 'code') == tmp_path / 'chosen'


def test_a_clerk_keeps_data_and_studies_in_the_workspace(
    tmp_path: pathlib.Path) -> None:
    clerk = amos.Clerk()
    assert clerk.root_folder == tmp_path
    assert clerk.input_folder == tmp_path / 'data'
    assert clerk.interim_folder == clerk.input_folder
    assert clerk.output_folder == tmp_path / 'studies'
    assert (tmp_path / 'data').is_dir()
    assert (tmp_path / 'studies').is_dir()


def test_a_data_subfolder_is_the_input_folder(tmp_path: pathlib.Path) -> None:
    clerk = amos.Clerk(data_subfolder = 'crime/nibrs')
    assert clerk.input_folder == tmp_path / 'data' / 'crime' / 'nibrs'
    assert clerk.interim_folder == clerk.input_folder
    assert clerk.input_folder.is_dir()
    clerk = amos.Clerk(
        root_folder = tmp_path / 'other', data_subfolder = 'crime',
        interim_folder = 'data/crime/cache')
    assert clerk.input_folder == tmp_path / 'other' / 'data' / 'crime'
    assert clerk.interim_folder == tmp_path / 'other' / 'data/crime/cache'


def test_a_project_has_a_clerk_of_its_files_section(
    tmp_path: pathlib.Path) -> None:
    settings = {
        'general': {'label': 'target', 'seed': SEED},
        'files': {'data_subfolder': 'survey'},
        'study_project': {'study_workers': 'analyst'},
        'analyst': {'techniques': 'stratified, logit'}}
    project = amos.Project.create(settings, item = make_numeric(), id = 'run')
    assert isinstance(project.clerk, amos.Clerk)
    assert project.clerk.input_folder == tmp_path / 'data' / 'survey'
    assert project.export() == tmp_path / 'studies' / 'run'
