"""Check notebook setup behavior without modifying the test runner's Python."""
import builtins
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


NOTEBOOKS = Path(__file__).resolve().parents[1] / "docs" / "notebooks"


def setup_cells(name="1_spatial_integration_imaging"):
    cells = json.loads((NOTEBOOKS / (name + ".ipynb")).read_text())["cells"]
    sources = {c.get("id"): "".join(c["source"]) for c in cells}
    return sources["fusemap-python-setup"], sources["fusemap-package-install"]


def environment(version, colab=True, gpu=False, pip_fails=False):
    events = []

    def run(command, check):
        assert check  # A failed installation must stop the cell.
        events.append(("pip", command))
        if pip_fails:
            raise subprocess.CalledProcessError(1, command)

    def graph(*args, **kwargs):
        events.append(("graph", kwargs))
        return object()

    modules = {
        "sys": SimpleNamespace(
            version_info=version,
            version=".".join(map(str, version)),
            executable="/test/kernel/python",
            modules={"google.colab": object()} if colab else {},
        ),
        "subprocess": SimpleNamespace(run=run),
        "condacolab": SimpleNamespace(
            install=lambda **kwargs: events.append(("switch_kernel", kwargs))
        ),
        "torch": SimpleNamespace(
            __version__="2.0.1+cu117", cuda=SimpleNamespace(is_available=lambda: gpu)
        ),
        "dgl": SimpleNamespace(__version__="1.1.1", graph=graph),
        "fusemap": SimpleNamespace(__version__="1.1.3"),
    }

    def import_module(name, *args, **kwargs):
        if name in modules:
            events.append(("import", name))
            return modules[name]
        return builtins.__import__(name, *args, **kwargs)

    namespace = {"__builtins__": dict(vars(builtins), __import__=import_module)}
    return namespace, events


def test_python_313_colab_prepares_actual_311_kernel():
    bootstrap, _ = setup_cells()
    namespace, events = environment((3, 13, 0))
    exec(bootstrap, namespace)
    switches = [value for kind, value in events if kind == "switch_kernel"]
    assert switches == [{"python_version": "3.11", "dependencies": {"numpy": "1.26.*", "matplotlib": "3.8.*"}}]
    commands = [value for kind, value in events if kind == "pip"]
    assert len(commands) == 1
    assert commands[0][:3] == ["/test/kernel/python", "-m", "pip"]
    assert "9df6578d7547f748e22d16b3a5755290bb41b9ad" in commands[0][-1]


@pytest.mark.parametrize("colab", [True, False])
def test_supported_kernel_is_not_replaced_on_rerun(colab):
    bootstrap, _ = setup_cells()
    namespace, events = environment((3, 11, 0), colab=colab)
    exec(bootstrap, namespace)
    exec(bootstrap, namespace)
    assert not any(kind in {"pip", "switch_kernel"} for kind, _ in events)


def test_unsupported_local_kernel_is_not_modified():
    bootstrap, _ = setup_cells()
    namespace, events = environment((3, 13, 0), colab=False)
    with pytest.raises(RuntimeError, match="Local notebooks require"):
        exec(bootstrap, namespace)
    assert not any(kind in {"pip", "switch_kernel"} for kind, _ in events)


def test_install_cannot_continue_before_kernel_restart():
    _, install = setup_cells()
    namespace, events = environment((3, 13, 0))
    with pytest.raises(RuntimeError, match="Wait for Colab to reconnect"):
        exec(install, namespace)
    assert not any(kind == "pip" for kind, _ in events)


@pytest.mark.parametrize("gpu", [False, True])
def test_install_uses_pinned_release_and_checks_gpu_backend(gpu):
    _, install = setup_cells()
    namespace, events = environment((3, 11, 0), gpu=gpu)
    exec(install, namespace)
    commands = [value for kind, value in events if kind == "pip"]
    assert "fusemap[tutorials]==1.1.3" in commands[0]
    assert len(commands) == (2 if gpu else 1)
    if gpu:
        assert "dgl==1.1.1+cu117" in commands[1]
        assert events.index(("pip", commands[1])) < events.index(("import", "fusemap"))
        assert ("graph", {"num_nodes": 2, "device": "cuda"}) in events
    else:
        assert not any(kind == "graph" for kind, _ in events)


def test_failed_install_does_not_import_fusemap():
    _, install = setup_cells()
    namespace, events = environment((3, 11, 0), pip_fails=True)
    with pytest.raises(subprocess.CalledProcessError):
        exec(install, namespace)
    assert ("import", "fusemap") not in events


def test_all_colab_notebooks_provide_the_same_kernel_setup():
    reference, _ = setup_cells()
    checked = 0
    for path in NOTEBOOKS.glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        if "colab.research.google.com/github/" not in "".join(notebook["cells"][0]["source"]):
            continue
        bootstrap, install = setup_cells(path.stem)
        assert bootstrap == reference, path
        assert "Wait for Colab to reconnect" in install, path
        namespace, events = environment((3, 13, 0))
        with pytest.raises(RuntimeError, match="Wait for Colab to reconnect"):
            exec(install, namespace)
        assert not any(kind == "pip" for kind, _ in events)
        checked += 1
    assert checked == 12
