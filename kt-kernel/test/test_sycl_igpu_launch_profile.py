"""Validate delivery shell settings without launching a model or a GPU."""
from pathlib import Path
import os
import subprocess

SCRIPT = Path(__file__).parents[1] / 'scripts/launch_sycl_igpu_qwen35b.sh'


def test_launch_profile_overrides_stale_settings_and_preserves_paths(tmp_path):
    model = tmp_path / 'model with spaces'; model.mkdir()
    setvars = tmp_path / 'setvars with spaces.sh'; setvars.write_text('export TEST_ONEAPI_LOADED=1\n')
    python = tmp_path / 'fake python'
    python.write_text('''#!/bin/bash
if [[ $1 == -c ]]; then exit 0; fi
printf '%s\\n' "$KT_SYCL_XMX_FUSED_EXPERT/$KT_SYCL_XMX_DECODE_MERGE/$KT_SYCL_XMX_GATE_SPLITS/$KT_SYCL_XMX_GRAPH/$KT_SYCL_XMX_TRACE/$TEST_ONEAPI_LOADED"
printf '<%s>\\n' "$@"
''')
    python.chmod(0o755)
    env = os.environ | {'KT_MODEL_PATH': str(model), 'KT_PYTHON': str(python), 'ONEAPI_SETVARS': str(setvars),
                        'KT_SYCL_XMX_FUSED_EXPERT': '0', 'KT_SYCL_XMX_DECODE_MERGE': 'fused',
                        'KT_SYCL_XMX_GRAPH': '1', 'KT_SYCL_XMX_TRACE': 'device'}
    result = subprocess.run(['bash', str(SCRIPT), '--log-level', 'warning'], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[0] == '1/separate/2/0/0/1'
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
