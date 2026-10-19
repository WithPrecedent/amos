"""Tests fitting models on a GPU, which the "use_gpu" setting chooses.

No test needs a GPU: the CUDA driver and the GPUs that `amos` finds are faked.
"""

from __future__ import annotations

import ctypes
import dataclasses
import json
import pathlib
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import sklearn.dummy
from conftest import SEED, make_numeric, requires

import amos
from amos import inferers, models, utilities

TITAN = {
    'name': 'NVIDIA TITAN Xp',
    'compute_capability': '6.1',
    'memory': 12.0,
    'cuda': '12.4'}


class _Driver:
    """Answers like NVIDIA's CUDA driver on a computer with one TITAN Xp."""

    def __init__(self, start: int = 0) -> None:
        self.start = start

    def cuInit(self, flags: int) -> int:  # noqa: N802
        return self.start

    def cuDeviceGetCount(self, count: Any) -> int:  # noqa: N802
        count._obj.value = 1
        return 0

    def cuDriverGetVersion(self, version: Any) -> int:  # noqa: N802
        version._obj.value = 12040
        return 0

    def cuDeviceGet(self, device: Any, number: int) -> int:  # noqa: N802
        device._obj.value = number
        return 0

    def cuDeviceGetName(self, name: Any, size: int, device: Any) -> int:  # noqa: N802
        name.value = TITAN['name'].encode()
        return 0

    def cuDeviceGetAttribute(  # noqa: N802
        self, value: Any, attribute: int, device: Any) -> int:
        value._obj.value = {75: 6, 76: 1}[attribute]
        return 0

    def cuDeviceTotalMem_v2(self, memory: Any, device: Any) -> int:  # noqa: N802
        memory._obj.value = 12 * 2**30
        return 0


def _check_device(device: str, fails: str | None) -> None:
    """Fails on the GPU if `fails` is "gpu", and anywhere if it is "always"."""
    if fails == 'always':
        raise ValueError('the data is wrong')
    if fails == 'gpu' and device == 'cuda':
        raise RuntimeError('no kernel image is available for the device')


class _OnDevice(sklearn.dummy.DummyClassifier):
    """A classifier that keeps its device and can fail on the GPU.

    It fails on the GPU if `fails` is "gpu", and on any device if it is
    "always".
    """

    def __init__(
        self,
        strategy: str = 'prior',
        device: str = 'cpu',
        fails: str | None = None) -> None:
        super().__init__(strategy = strategy)
        self.device = device
        self.fails = fails

    def fit(self, x: Any, y: Any) -> _OnDevice:
        _check_device(self.device, self.fails)
        return super().fit(x, y)


class _OnDeviceRegressor(sklearn.dummy.DummyRegressor):
    """A regressor that keeps its device and can fail on the GPU, like
    `_OnDevice`."""

    def __init__(
        self,
        strategy: str = 'mean',
        device: str = 'cpu',
        fails: str | None = None) -> None:
        super().__init__(strategy = strategy)
        self.device = device
        self.fails = fails

    def fit(self, x: Any, y: Any) -> _OnDeviceRegressor:
        _check_device(self.device, self.fails)
        return super().fit(x, y)


@pytest.fixture(autouse = True)
def _forget_gpus(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Runs each test in its own folder, looking for GPUs again."""
    monkeypatch.chdir(tmp_path)
    # A test may replace `gpus`, so the one whose answer is kept is saved.
    found = utilities.gpus
    found.cache_clear()
    yield
    found.cache_clear()


@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes `amos` find a TITAN Xp."""
    monkeypatch.setattr(utilities, 'gpus', lambda: (dict(TITAN),))


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes `amos` find no GPU."""
    monkeypatch.setattr(utilities, 'gpus', tuple)


@pytest.fixture
def on_device() -> type[amos.Model]:
    """A model of `_OnDevice`, which can be fitted on a GPU."""

    @dataclasses.dataclass
    class OnDevice(amos.Model):
        tools: ClassVar[dict[str, Any]] = {'classify': _OnDevice}

        def _device_parameters(self, *, gpu: bool) -> dict[str, Any] | None:
            return {'device': 'cuda' if gpu else 'cpu'}

    return OnDevice


@pytest.fixture
def learners() -> None:
    """Adds models that inferers can use as learners on a GPU.

    "device_learner" classifies and regresses on a GPU, and
    "failing_learner" does too, but fails on the GPU.
    """

    @dataclasses.dataclass
    class DeviceLearner(amos.Model):
        tools: ClassVar[dict[str, Any]] = {
            'classify': _OnDevice, 'regress': _OnDeviceRegressor}

        def _device_parameters(self, *, gpu: bool) -> dict[str, Any] | None:
            return {'device': 'cuda' if gpu else 'cpu'}

    @dataclasses.dataclass
    class FailingLearner(DeviceLearner):
        parameters: dict[str, Any] = dataclasses.field(
            default_factory = lambda: {'fails': 'gpu'})


def _dataset(use_gpu: bool = True) -> amos.Dataset:
    """Returns a split dataset of numbers."""
    dataset = amos.Dataset(
        make_numeric(), label = 'target', seed = SEED, use_gpu = use_gpu)
    return amos.splitters.Stratified().apply(dataset)


""" Finding GPUs """


def test_gpus_are_found_with_the_cuda_driver(
    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utilities, '_cuda_driver', _Driver)
    assert utilities.gpus() == (TITAN,)


def test_no_driver_means_no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utilities, '_cuda_driver', lambda: None)
    assert utilities.gpus() == ()


def test_a_driver_without_a_gpu_means_no_gpus(
    monkeypatch: pytest.MonkeyPatch) -> None:
    # 100 is CUDA_ERROR_NO_DEVICE.
    monkeypatch.setattr(utilities, '_cuda_driver', lambda: _Driver(start = 100))
    assert utilities.gpus() == ()


def test_the_driver_is_loaded_if_it_is_installed(
    monkeypatch: pytest.MonkeyPatch) -> None:
    def load(name: str) -> str:
        if name != 'nvcuda.dll':
            raise OSError(name)
        return name

    def missing(name: str) -> str:
        raise OSError(name)

    monkeypatch.setattr(ctypes, 'CDLL', load)
    assert utilities._cuda_driver() == 'nvcuda.dll'
    monkeypatch.setattr(ctypes, 'CDLL', missing)
    assert utilities._cuda_driver() is None


def test_the_environment_describes_gpus_when_asked(gpu: None) -> None:
    assert 'gpus' not in utilities.environment()
    assert utilities.environment(gpu = True)['gpus'] == [TITAN]


""" The use_gpu setting """


def test_use_gpu_is_false_by_default() -> None:
    assert amos.Dataset().use_gpu is False
    assert amos.Dataset.create(make_numeric()).use_gpu is False


@pytest.mark.parametrize(('value', 'expected'), [
    (True, True), (False, False), (None, False), ('true', True),
    ('False', False)])
def test_use_gpu_is_true_or_false(value: Any, expected: bool) -> None:
    assert amos.Dataset(use_gpu = value).use_gpu is expected


@pytest.mark.parametrize('value', ['yes', 1, 'cuda'])
def test_use_gpu_must_be_true_or_false(value: Any) -> None:
    with pytest.raises(TypeError, match = 'use_gpu must be true or false'):
        amos.Dataset(use_gpu = value)


def test_create_sets_use_gpu_only_if_it_is_passed() -> None:
    dataset = amos.Dataset(use_gpu = True)
    assert amos.Dataset.create(dataset).use_gpu is True
    assert amos.Dataset.create(dataset, use_gpu = 'false').use_gpu is False


def test_the_repr_says_when_the_gpu_is_used() -> None:
    assert 'use_gpu' not in repr(amos.Dataset())
    assert 'use_gpu=True' in repr(amos.Dataset(use_gpu = True))


""" Fitting models """


def test_models_are_fitted_on_the_cpu_without_use_gpu(
    gpu: None,
    on_device: type[amos.Model]) -> None:
    dataset = on_device().apply(_dataset(use_gpu = False))
    assert dataset.model.device == 'cpu'
    assert dataset.history[-1]['parameters']['device'] == 'cpu'


def test_use_gpu_fits_models_on_the_gpu(
    gpu: None,
    on_device: type[amos.Model]) -> None:
    dataset = on_device().apply(_dataset())
    assert dataset.model.device == 'cuda'
    assert dataset.history[-1]['parameters']['device'] == 'cuda'


def test_a_search_is_on_the_gpu(
    gpu: None,
    on_device: type[amos.Model]) -> None:
    dataset = on_device().apply(
        _dataset(), search = 'grid', strategy = ['prior', 'stratified'], cv = 3)
    assert dataset.model.device == 'cuda'
    assert len(dataset.tables['on_device_search']) == 2


def test_a_model_without_a_gpu_is_fitted_on_the_cpu(
    no_gpu: None,
    on_device: type[amos.Model]) -> None:
    with pytest.warns(UserWarning, match = 'no NVIDIA GPU was found'):
        dataset = on_device().apply(_dataset())
    assert dataset.model.device == 'cpu'


def test_a_model_that_fails_on_the_gpu_is_fitted_on_the_cpu(
    gpu: None,
    on_device: type[amos.Model]) -> None:
    with pytest.warns(UserWarning, match = 'could not be fitted on the GPU'):
        dataset = on_device().apply(_dataset(), fails = 'gpu')
    assert dataset.model.device == 'cpu'
    assert dataset.history[-1]['parameters']['device'] == 'cpu'
    assert len(dataset.predictions) == len(dataset.test)


def test_other_errors_are_raised(
    gpu: None,
    on_device: type[amos.Model]) -> None:
    with (
        pytest.warns(UserWarning, match = 'could not be fitted on the GPU'),
        pytest.raises(ValueError, match = 'the data is wrong')):
        on_device().apply(_dataset(), fails = 'always')
    with pytest.raises(ValueError, match = 'the data is wrong'):
        on_device().apply(_dataset(use_gpu = False), fails = 'always')


def test_models_that_only_use_the_cpu_do_not_look_for_a_gpu(
    monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        pytest.fail('looked for a GPU')

    monkeypatch.setattr(utilities, 'gpus', fail)
    dataset = models.Logit().apply(_dataset())
    assert dataset.model is not None


def test_the_parameters_of_a_model_take_precedence(
    gpu: None,
    on_device: type[amos.Model]) -> None:
    dataset = on_device().apply(_dataset(), device = 'cpu')
    assert dataset.model.device == 'cpu'


@pytest.mark.parametrize(('kind', 'on_gpu', 'on_cpu'), [
    (models.Catboost,
     {'task_type': 'GPU', 'devices': '0'},
     {'task_type': 'CPU'}),
    (models.Lightgbm, {'device_type': 'gpu'}, {'device_type': 'cpu'}),
    (models.Tabpfn, {'device': 'cuda'}, {'device': 'cpu'}),
    (models.Xgboost, {'device': 'cuda'}, {'device': 'cpu'}),
    (models.RandomForest, None, None)])
def test_the_device_parameters_of_each_model(
    kind: type[amos.Model],
    on_gpu: dict[str, str] | None,
    on_cpu: dict[str, str] | None) -> None:
    assert kind()._device_parameters(gpu = True) == on_gpu
    assert kind()._device_parameters(gpu = False) == on_cpu


@pytest.mark.parametrize(('kind', 'package', 'name', 'value'), [
    (models.Catboost, 'catboost', 'task_type', 'CPU'),
    (models.Lightgbm, 'lightgbm', 'device_type', 'cpu'),
    (models.Xgboost, 'xgboost', 'device', 'cpu')])
def test_models_are_built_for_the_cpu(
    kind: type[amos.Model],
    package: str,
    name: str,
    value: str) -> None:
    requires(package)
    dataset = kind().apply(_dataset(use_gpu = False), n_estimators = 5)
    assert dataset.model.get_params()[name] == value


""" Inferers """


def _treated(use_gpu: bool = True) -> amos.Dataset:
    """Returns a dataset in which a treatment with two values raises the label."""
    rng = np.random.default_rng(11)
    need = rng.normal(size = 200)
    treated = (rng.random(200) < 0.5).astype(int)
    data = pd.DataFrame({
        'need': need,
        'treated': treated,
        'outcome': 2 * treated + need + rng.normal(size = 200)})
    return amos.Dataset(data, label = 'outcome', seed = SEED, use_gpu = use_gpu)


def _partially_linear(
    dataset: amos.Dataset,
    outcome_model: str = 'device_learner',
    treatment_model: str | None = None) -> dict[str, str]:
    """Applies DoubleML's `partially_linear` and returns its learners' devices."""
    requires('doubleml')
    inferers.PartiallyLinear().apply(
        dataset, treatment = 'treated', outcome_model = outcome_model,
        treatment_model = treatment_model, n_folds = 2)
    assert 'partially_linear' in dataset.tables
    learned = dataset.fitted['partially_linear'].learner
    return {
        role: getattr(learned[role], 'device', 'none') for role in ('ml_l', 'ml_m')}


def test_inferers_fit_their_learners_on_the_gpu(learners: None, gpu: None) -> None:
    devices = _partially_linear(_treated())
    assert devices == {'ml_l': 'cuda', 'ml_m': 'cuda'}


def test_inferers_fit_their_learners_on_the_cpu_without_use_gpu(
    learners: None,
    monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        pytest.fail('looked for a GPU')

    monkeypatch.setattr(utilities, 'gpus', fail)
    devices = _partially_linear(_treated(use_gpu = False))
    assert devices == {'ml_l': 'cpu', 'ml_m': 'cpu'}


def test_inferers_without_a_gpu_fit_their_learners_on_the_cpu(
    learners: None,
    no_gpu: None) -> None:
    with pytest.warns(UserWarning, match = 'no NVIDIA GPU was found'):
        devices = _partially_linear(_treated())
    assert devices == {'ml_l': 'cpu', 'ml_m': 'cpu'}


def test_a_learner_that_fails_on_the_gpu_is_fitted_on_the_cpu(
    learners: None,
    gpu: None) -> None:
    with pytest.warns(UserWarning) as warned:
        devices = _partially_linear(_treated(), outcome_model = 'failing_learner')
    assert devices == {'ml_l': 'cpu', 'ml_m': 'cpu'}
    # The model of both the label and the treatment is checked once.
    messages = [
        str(w.message) for w in warned
        if 'could not be fitted on the GPU' in str(w.message)]
    assert len(messages) == 1
    assert "'partially_linear' fits it on the CPU" in messages[0]
    assert 'no kernel image' in messages[0]


def test_only_learners_that_can_use_the_gpu_are_on_it(
    learners: None,
    gpu: None) -> None:
    with pytest.warns(UserWarning, match = 'could not be fitted on the GPU'):
        devices = _partially_linear(
            _treated(), outcome_model = 'failing_learner',
            treatment_model = 'device_learner')
    assert devices == {'ml_l': 'cpu', 'ml_m': 'cuda'}
    devices = _partially_linear(_treated(), treatment_model = 'logit')
    assert devices == {'ml_l': 'cuda', 'ml_m': 'none'}


def test_inferers_with_learners_for_the_cpu_do_not_look_for_a_gpu(
    monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        pytest.fail('looked for a GPU')

    monkeypatch.setattr(utilities, 'gpus', fail)
    devices = _partially_linear(
        _treated(), outcome_model = 'linear', treatment_model = 'logit')
    assert devices == {'ml_l': 'none', 'ml_m': 'none'}


def test_causalml_fits_its_learners_on_the_gpu(learners: None, gpu: None) -> None:
    requires('causalml')
    dataset = _treated()
    inferers.XLearner().apply(
        dataset, treatment = 'treated', outcome_model = 'device_learner',
        n_folds = 2)
    estimator = dataset.fitted['x_learner']
    assert estimator.model_mu_c.device == 'cuda'
    assert 'x_learner' in dataset.tables


def test_a_learner_is_checked_with_a_task_it_can_do(
    on_device: type[amos.Model],
    gpu: None) -> None:
    # `on_device` only classifies.
    assert inferers._fits_on_gpu('on_device', 'regress', SEED, 'test')


@pytest.mark.parametrize(('package', 'name', 'on_gpu', 'on_cpu'), [
    ('catboost', 'task_type', 'GPU', 'CPU'),
    ('lightgbm', 'device_type', 'gpu', 'cpu'),
    ('tabpfn', 'device', 'cuda', 'cpu'),
    ('xgboost', 'device', 'cuda', 'cpu')])
def test_learners_are_built_for_their_device(
    package: str,
    name: str,
    on_gpu: str,
    on_cpu: str) -> None:
    requires(package)
    for task in ('classify', 'regress'):
        learner = inferers._learner(package, task, SEED, gpu = True)
        assert learner.get_params()[name] == on_gpu
        learner = inferers._learner(package, task, SEED)
        assert learner.get_params()[name] == on_cpu


""" Projects """


def _settings(**general: Any) -> dict[str, Any]:
    """Returns settings for a small project with "general" settings."""
    return {
        'general': {'label': 'target', 'seed': SEED, **general},
        'study_project': {'study_workers': 'analyst'},
        'analyst': {
            'steps': 'split, model',
            'split_techniques': 'stratified',
            'model_techniques': 'logit'}}


def test_projects_use_gpu_only_if_the_settings_say_so() -> None:
    project = amos.Project.create(_settings(), item = make_numeric())
    assert project.result.use_gpu is False
    project = amos.Project.create(
        _settings(use_gpu = True), item = make_numeric())
    assert project.result.use_gpu is True


def test_use_gpu_can_be_set_in_a_settings_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / 'settings.ini'
    path.write_text(
        '[general]\nlabel = target\nuse_gpu = true\n\n'
        '[study_project]\nstudy_workers = analyst\n\n'
        '[analyst]\nsteps = split, model\nsplit_techniques = stratified\n'
        'model_techniques = logit\n',
        encoding = 'utf-8')
    project = amos.Project.create(path, item = make_numeric())
    assert project.result.use_gpu is True


def test_the_settings_decide_for_a_dataset_too() -> None:
    dataset = amos.Dataset.create(make_numeric(), use_gpu = True)
    project = amos.Project.create(_settings(), item = dataset)
    assert project.result.use_gpu is True
    project = amos.Project.create(_settings(use_gpu = False), item = dataset)
    assert project.result.use_gpu is False


def test_a_loader_keeps_use_gpu() -> None:
    make_numeric().to_csv('cases.csv', index = False)
    settings = _settings(use_gpu = True)
    settings['study_project']['study_workers'] = 'wrangler, analyst'
    settings['wrangler'] = {'techniques': 'load_file'}
    settings['load_file_parameters'] = {'source': 'cases.csv'}
    project = amos.Project.create(settings)
    assert project.result.use_gpu is True


def test_export_describes_the_gpus(gpu: None) -> None:
    project = amos.Project.create(
        _settings(use_gpu = True), item = make_numeric(), id = 'run')
    folder = project.export()
    described = json.loads((folder / 'environment.json').read_text('utf-8'))
    assert described['gpus'] == [TITAN]
    project = amos.Project.create(_settings(), item = make_numeric(), id = 'cpu')
    described = json.loads(
        (project.export() / 'environment.json').read_text('utf-8'))
    assert 'gpus' not in described
