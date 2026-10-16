"""Validate delivery shell settings without launching a model or a GPU."""
from pathlib import Path
import os
import subprocess
import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/launch_sycl_igpu_qwen35b.sh'


@pytest.mark.parametrize('preset,expected', [('v5', 'block/scalar'), ('v6', 'gather/vector')])
def test_launch_profile_overrides_stale_settings_and_preserves_paths(tmp_path, preset, expected):
    model = tmp_path / 'model with spaces'; model.mkdir()
    setvars = tmp_path / 'setvars with spaces.sh'; setvars.write_text('export TEST_ONEAPI_LOADED=1\n')
    python = tmp_path / 'fake python'
    python.write_text('''#!/bin/bash
if [[ $1 == -c ]]; then exit 0; fi
printf '%s\\n' "$KT_SYCL_XMX_FUSED_EXPERT/$KT_SYCL_XMX_DECODE_MERGE/$KT_SYCL_XMX_GATE_SPLITS/$KT_SYCL_XMX_GRAPH/$KT_SYCL_XMX_TRACE/$TEST_ONEAPI_LOADED/$KT_SYCL_XMX_PREFILL_ROWS/$KT_SYCL_XMX_PREFILL_DOWN_ROWS/$KT_SYCL_XMX_PREFILL_N/$KT_SYCL_XMX_PREFILL_LOAD/$KT_SYCL_XMX_PREFILL_MERGE"
printf '<%s>\\n' "$@"
''')
    python.chmod(0o755)
    env = os.environ | {'KT_MODEL_PATH': str(model), 'KT_PYTHON': str(python), 'ONEAPI_SETVARS': str(setvars),
                        'KT_IGPU_PREFILL_PROFILE': preset, 'KT_SYCL_XMX_FUSED_EXPERT': '0', 'KT_SYCL_XMX_DECODE_MERGE': 'fused',
                        'KT_SYCL_XMX_GRAPH': '1', 'KT_SYCL_XMX_TRACE': 'device',
                        'KT_SYCL_XMX_PREFILL_ROWS': '32', 'KT_SYCL_XMX_PREFILL_DOWN_ROWS': '8', 'KT_SYCL_XMX_PREFILL_N': '16', 'KT_SYCL_XMX_PREFILL_LOAD': 'block', 'KT_SYCL_XMX_PREFILL_MERGE': 'scalar'}
    result = subprocess.run(['bash', str(SCRIPT), '--log-level', 'warning'], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[0] == '1/separate/2/0/0/1/8/16/32/' + expected
    assert f'<{model}>' in result.stdout
    assert '<--log-level>\n<warning>' in result.stdout


def test_missing_model_is_an_error(tmp_path):
    env = os.environ | {'KT_MODEL_PATH': str(tmp_path / 'missing')}
    result = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Model directory not found' in result.stderr



def test_old_extension_preflight_is_not_ignored(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    setvars = tmp_path / 'setvars.sh'; setvars.write_text('true\n')
    python = tmp_path / 'python'; python.write_text('#!/bin/bash\nexit 7\n'); python.chmod(0o755)
    env = os.environ | {'KT_MODEL_PATH': str(model), 'KT_PYTHON': str(python), 'ONEAPI_SETVARS': str(setvars)}
    result = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 7


def test_invalid_prefill_profile_is_an_error():
    env = os.environ | {'KT_IGPU_PREFILL_PROFILE': 'invalid'}
    result = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and 'KT_IGPU_PREFILL_PROFILE must be v5 or v6' in result.stderr


def test_default_profile_uses_validated_vectorized_prefill(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    setvars = tmp_path / 'setvars.sh'; setvars.write_text('true\n')
    python = tmp_path / 'python'
    python.write_text('''#!/bin/bash
if [[ $1 == -c ]]; then
  [[ $2 == *">= 6"* ]] || exit 8
  exit 0
fi
printf '%s/%s\n' "$KT_SYCL_XMX_PREFILL_LOAD" "$KT_SYCL_XMX_PREFILL_MERGE"
''')
    python.chmod(0o755)
    env = os.environ | {'KT_MODEL_PATH': str(model), 'KT_PYTHON': str(python),
                        'ONEAPI_SETVARS': str(setvars)}
    env.pop('KT_IGPU_PREFILL_PROFILE', None)
    result = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'gather/vector'


def test_invalid_cuda_dense_profile_is_an_error():
    env = os.environ | {'KT_IGPU_DENSE_PROFILE': 'typo'}
    result = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'KT_IGPU_DENSE_PROFILE must be' in result.stderr


@pytest.mark.parametrize('profile', ['native', 'linear', 'head-pad4', 'both-row', 'both-pad4'])
def test_cuda_dense_profile_is_exported_and_preflighted(tmp_path, profile):
    model = tmp_path / 'model'; model.mkdir()
    setvars = tmp_path / 'setvars.sh'; setvars.write_text('true\n')
    python = tmp_path / 'python'
    python.write_text('''#!/bin/bash
if [[ $1 == -c ]]; then
  if [[ $2 == *kt_cuda_dense* ]]; then echo dense-preflight; fi
  exit 0
fi
printf 'mode=%s\n' "$SGLANG_KT_CUDA_DENSE"
''')
    python.chmod(0o755)
    env = os.environ | {'KT_MODEL_PATH': str(model), 'KT_PYTHON': str(python),
                        'ONEAPI_SETVARS': str(setvars), 'KT_IGPU_DENSE_PROFILE': profile,
                        'SGLANG_KT_CUDA_DENSE': 'stale-inherited-value'}
    result = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True, check=True)
    assert ('dense-preflight' in result.stdout) == (profile != 'native')
    assert f'mode={profile}' in result.stdout


def test_old_sglang_dense_preflight_is_not_ignored(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    setvars = tmp_path / 'setvars.sh'; setvars.write_text('true\n')
    python = tmp_path / 'python'
    python.write_text('''#!/bin/bash
if [[ $1 == -c && $2 == *kt_cuda_dense* ]]; then exit 9; fi
if [[ $1 == -m ]]; then echo unexpected-launch; fi
''')
    python.chmod(0o755)
    env = os.environ | {'KT_MODEL_PATH': str(model), 'KT_PYTHON': str(python),
                        'ONEAPI_SETVARS': str(setvars), 'KT_IGPU_DENSE_PROFILE': 'linear'}
    result = subprocess.run(['bash', str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 9
    assert 'unexpected-launch' not in result.stdout
