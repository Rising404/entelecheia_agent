"""Entry ingress 归包后的所有权与冷导入围栏。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PERSONAGRAPH_ROOT = Path(__file__).resolve().parents[4] / "src" / "personagraph"
INGRESS_ROOT = PERSONAGRAPH_ROOT / "runtime" / "entry" / "ingress"


def test_entry_ingress_has_one_explicit_owner_per_responsibility() -> None:
    assert (INGRESS_ROOT / "contracts.py").is_file()
    assert (INGRESS_ROOT / "model.py").is_file()
    assert (INGRESS_ROOT / "model_contracts.py").is_file()
    assert (INGRESS_ROOT / "policy.py").is_file()
    assert (INGRESS_ROOT / "classification.py").is_file()


def test_retired_flat_entry_ingress_modules_are_absent() -> None:
    entry_root = INGRESS_ROOT.parent

    assert not (entry_root / "classification_stage.py").exists()
    assert not (entry_root / "ingress_contracts.py").exists()
    assert (
        importlib.util.find_spec("personagraph.runtime.entry.classification_stage")
        is None
    )
    assert (
        importlib.util.find_spec("personagraph.runtime.entry.ingress_contracts") is None
    )


def test_importing_entry_ingress_package_does_not_load_its_implementation_modules() -> (
    None
):
    code = """
import importlib
import json
import sys

importlib.import_module('personagraph.runtime.entry.ingress')
loaded = sorted(
    name
    for name in sys.modules
    if name in {
        'personagraph.runtime.entry.ingress.classification',
        'personagraph.runtime.entry.ingress.contracts',
        'personagraph.runtime.entry.ingress.model',
        'personagraph.runtime.entry.ingress.model_contracts',
        'personagraph.runtime.entry.ingress.policy',
    }
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_l1_model_contract_does_not_initialize_l2() -> None:
    code = """
import importlib.abc
import sys

class RejectL2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'personagraph.l2' or fullname.startswith('personagraph.l2.'):
            raise AssertionError(f'cold L1 model contract imported {fullname}')
        return None

sys.meta_path.insert(0, RejectL2())
from personagraph.runtime.entry.ingress.model_contracts import EntryClassification

classification = EntryClassification.model_validate({
    'processing_level': 'L1',
    'task_matches': [],
})
print(classification.processing_level)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "L1"
