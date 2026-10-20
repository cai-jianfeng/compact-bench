# Task Preservation environment recipes

The HF dataset's `e1/instances.json` supplies each public repository, base commit,
problem statement, patches, and test lists. The runner downloads the corresponding
source archive from GitHub and prepares the environment automatically.

1. Create a separate environment for each instance. The runner accepts an explicit Python executable through `CB_PYTHON`; otherwise it uses the runner's interpreter. Historical repositories that fail to import on Python 3.10+ use Python 3.9. Later environments used the system Python 3.12. These are the documented interpreter choices; this supplement does not contain a per-run environment lock.
2. For pytest source archives, set `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST=7.99.0` before building/installing the repository and while running it. This is the “setuptools_scm shim” mentioned in the instance table, not a source-code patch.
3. Apply the instance's test patch to the base source tree and check that FAIL_TO_PASS fails before continuation. Restore the frozen working-directory snapshot for each continuation. The gold patch is retained as metadata and is not applied by the evaluation runner.
4. Install the repository-specific prerequisites below, then install the repository non-editable, then install the shared constraints `pytest<8`, `werkzeug<3`, and `urllib3<2`. Reinstall repository-specific prerequisites with `--no-deps --force-reinstall` afterwards so repository installation cannot silently upgrade them.
5. The agent shell uses the instance environment's `bin` directory first on `PATH` and the live repository directory on `PYTHONPATH`. Command output returns the last 3,000 characters of stdout plus stderr. `read_file` returns the first 8,000 characters. Tool calls have a 120-second command timeout; judging allows 300 seconds per test invocation.
6. Before judging, install the current working tree into the environment with `pip install -q --no-deps --force-reinstall <working-tree>`. Run each normalized FAIL_TO_PASS node separately as shown below. A nonzero test return code fails that arm. The implemented endpoint checks FAIL_TO_PASS; PASS_TO_PASS identifiers are retained as public metadata and are not part of this judge.

| Repository | Additional installation constraints |
| --- | --- |
| xarray | `setuptools_scm<8`, `numpy<2`, `pandas<2` |
| pylint | `py` |
| sphinx | `docutils<0.18`, `sphinxcontrib-applehelp==1.0.2`, `sphinxcontrib-devhelp==1.0.2`, `sphinxcontrib-htmlhelp==2.0.0`, `sphinxcontrib-qthelp==1.0.3`, `sphinxcontrib-serializinghtml==1.1.5` |
| pytest | The source-archive version environment variable above |
| requests, flask, sympy | Shared constraints; Python 3.9 where required by the source version |

The judge's test command is:

```bash
python -m pytest -o filterwarnings=ignore::DeprecationWarning -x -q TEST_NODE_ID
```

For bare function names in SymPy FAIL_TO_PASS metadata, locate the function definition in the Python files touched by the test patch and form `path/to/test_file.py::test_name`. If no definition is found, the historical runner falls back to the first Python test-patch file; this is recorded as part of the implementation, not recommended as a general resolver.

The working-directory snapshot at the end of phase A is restored before each counterfactual arm. Phase A and both arms use the same repository path. Frozen histories and snapshots are downloaded from the HF dataset and restored together. New model outputs remain stochastic. Repository constraints are not full transitive locks; their later resolution can differ from the historical environments.
