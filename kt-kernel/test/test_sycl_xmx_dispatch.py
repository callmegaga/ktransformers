"""Backend selection tests without importing GPU extensions or torch."""
import ast
import os
from pathlib import Path
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).parents[1] / 'python/utils/amx.py'


def select(legacy=object(), xmx=object()):
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_select_sycl_int4_backend')
    scope = {'os': os, 'SYCLGPTQInt4_MOE': legacy, 'SYCLXMXGPTQInt4_MOE': xmx}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['_select_sycl_int4_backend']


def test_default_remains_legacy():
    legacy = object()
    with patch.dict(os.environ, {}, clear=True):
        assert select(legacy=legacy)() is legacy


def test_xmx_is_explicit_and_normalized():
    xmx = object()
    with patch.dict(os.environ, {'KT_SYCL_INT4_BACKEND': ' XMX '}):
        assert select(xmx=xmx)() is xmx


@pytest.mark.parametrize('value', ['auto', 'typo', ''])
def test_invalid_selection_is_not_a_fallback(value):
    with patch.dict(os.environ, {'KT_SYCL_INT4_BACKEND': value}):
        with pytest.raises(ValueError):
            select()()


def test_uncompiled_xmx_is_not_a_fallback():
    with patch.dict(os.environ, {'KT_SYCL_INT4_BACKEND': 'xmx'}):
        with pytest.raises(RuntimeError, match='not compiled'):
            select(xmx=None)()
