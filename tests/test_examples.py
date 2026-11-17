"""Прогон всех примеры.py: каждый файл — набор assert к своей лекции.

Запуск:  python -m pytest -q -n auto tests/
Каждый пример исполняется отдельным процессом во временном каталоге.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = sorted(ROOT.glob("Лекции/Часть_*/*/примеры.py"))


def test_examples_found():
    assert len(EXAMPLES) == 32


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.parent.name)
def test_example(path, tmp_path):
    r = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp_path,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, (r.stdout[-3000:] + "\n" + r.stderr[-3000:])
