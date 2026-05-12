'''
Shared pytest fixtures for the Basketball Play Analysis test suite.

Sets pipeline env vars to writable tmp paths *before* any pipeline module is
imported, exposes helpers to load top-level functions from files that have
heavy import-time side effects (Object_Tracking.py, simulation scripts), and
provides synthetic dataframes used by the quick suite.
'''


import ast
import os
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / 'fixtures'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Env / module-level setup -- runs once per pytest session before any imports
# of pipeline code happen inside test files.
# ---------------------------------------------------------------------------

_SESSION_TMP = pathlib.Path(os.environ.get('PYTEST_SESSION_TMP', '/tmp/bbpa_test_session'))
_SESSION_TMP.mkdir(parents=True, exist_ok=True)

os.environ.setdefault('LOG_DIR', str(_SESSION_TMP / 'logs'))
os.environ.setdefault('OUTPUT_DIR', str(_SESSION_TMP / 'output'))
os.environ.setdefault('MODEL_DIR', str(_SESSION_TMP / 'models'))
os.environ.setdefault('TRACKING_DIR', str(_SESSION_TMP / 'tracking'))
os.environ.setdefault('VIDEO_DIR', str(_SESSION_TMP / 'simulations'))
os.environ.setdefault('ASSETS_DIR', str(REPO_ROOT / 'assets'))
os.environ.setdefault('DeepSORT_DIR', str(REPO_ROOT / 'deep_sort'))

for _key in ('LOG_DIR', 'OUTPUT_DIR', 'MODEL_DIR', 'TRACKING_DIR', 'VIDEO_DIR'):
    pathlib.Path(os.environ[_key]).mkdir(parents=True, exist_ok=True)

# utils.configure_logger reads a module-level `log_dir` symbol; expose it.
import utils  # noqa: E402
utils.log_dir = os.environ['LOG_DIR']


# Several pipeline modules only define their module-level `logger` symbol
# inside `if __name__ == '__main__':`. When their functions are called
# directly from tests and an except branch fires, they NameError on `logger`
# before they can re-raise the real exception. Inject a real logger here so
# the original exception surfaces.
def _inject_module_loggers():
    import logging
    test_logger = logging.getLogger('bbpa_test')
    test_logger.addHandler(logging.NullHandler())

    # Feature_Engineering has heavy import-time side effects (mkdir on
    # /app/...) but no main() run on import -- safe to import here.
    try:
        import Feature_Engineering  # noqa: F401
        Feature_Engineering.logger = test_logger
    except Exception:
        pass


_inject_module_loggers()


# ---------------------------------------------------------------------------
# Source-extraction helper -- used to test functions defined in modules whose
# top-level body runs heavy side effects (cv2.VideoCapture, YOLO model load,
# pygame.init, ...). We grab just the named function via AST and exec it in
# an isolated namespace.
# ---------------------------------------------------------------------------

def load_callable_from_source(path, name, extra_globals=None):
    '''
    Compile and return the top-level function/class named `name` from `path`
    without executing the rest of the module.

    Parameters:
    [str|Path] path - Path to the source file
    [str] name - Name of the function or class to extract
    [dict] extra_globals - Names to inject into the exec namespace

    Returns:
    [callable] The extracted function or class
    '''
    source = pathlib.Path(path).read_text()
    tree = ast.parse(source)
    target_node = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            target_node = node
            break
    if target_node is None:
        raise LookupError(f"Symbol {name!r} not found at top level of {path}")

    mod = ast.Module(body=[target_node], type_ignores=[])
    namespace = {'__name__': f'_extracted_{name}'}
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(mod, str(path), 'exec'), namespace)
    return namespace[name]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope='session')
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def tmp_pipeline_dirs(tmp_path, monkeypatch):
    '''
    Per-test tmp directory tree mirroring the pipeline's expected env-var
    layout. Use this for extensive tests that want isolation.
    '''
    layout = {}
    for key in ('LOG_DIR', 'OUTPUT_DIR', 'MODEL_DIR', 'TRACKING_DIR', 'VIDEO_DIR'):
        sub = tmp_path / key.lower()
        sub.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(sub))
        layout[key] = sub
    monkeypatch.setenv('ASSETS_DIR', str(REPO_ROOT / 'assets'))
    layout['ASSETS_DIR'] = REPO_ROOT / 'assets'
    return layout


@pytest.fixture
def sample_tracking_df():
    '''Small dataframe matching the schema produced by Object_Tracking.py.'''
    rows = []
    track_ids = [0, 1, 2]
    class_for_track = {0: 'Basketball', 1: 'Team_A', 2: 'Team_B'}
    for frame in range(10):
        for tid in track_ids:
            mean = f"[{100.0 + frame} {200.0 + frame} 0.5 50.0 1.{frame} 2.{frame} 0.0 0.0]"
            cov = ("[[1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]\n"
                   " [0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0]\n"
                   " [0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0]\n"
                   " [0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0]\n"
                   " [0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0]\n"
                   " [0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]\n"
                   " [0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0]\n"
                   " [0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]]")
            rows.append({
                'TrackID': tid,
                'ClassID': class_for_track[tid],
                'Mean': mean,
                'Co-Variance': cov,
                'ConfidenceScore': 0.9,
                'State': 2,
                'Hits': frame + 1,
                'Age': frame + 1,
                'Features': '[0.1, 0.2, 0.3, 0.4]',
                'Frame': frame,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_tracking_csv(tmp_path, sample_tracking_df):
    path = tmp_path / 'detected_objects.csv'
    sample_tracking_df.to_csv(path)
    return path


@pytest.fixture
def sample_tracking_df_with_index(sample_tracking_df):
    '''Variant that includes the `Unnamed: 0` column that the real pipeline
    produces by writing then re-reading via to_csv/read_csv.'''
    df = sample_tracking_df.copy()
    df.insert(0, 'Unnamed: 0', range(len(df)))
    return df
