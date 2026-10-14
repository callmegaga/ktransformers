"""Opt-in integration checks: source oneAPI and set KT_TEST_SYCL_XMX=1."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KT_TEST_SYCL_XMX') != '1', reason='requires Intel XMX GPU and SYCL extension')


@pytest.fixture
def setup(monkeypatch):
    import torch
    from kt_kernel import kt_kernel_ext as ext
    monkeypatch.setenv('KT_SYCL_XMX_MODE', 'int8')
    cpu = ext.CPUInfer(2)
    mask = torch.zeros(4, dtype=torch.bool)
    config = ext.moe.MOEConfig(4, 2, 128, 128, mask.data_ptr())
    config.pool = cpu.backend_
    config.max_len = 17
    config.quant_config.bits = 4
    config.quant_config.group_size = 128
    config.quant_config.zero_point = False
    qw = torch.full((4, 16, 128), -2004318072, dtype=torch.int32)  # all GPTQ nibbles are 8, hence zero
    scales = torch.ones(4, 1, 128)
    for name in ('gate', 'up', 'down'):
        setattr(config, name + '_projs', [[t.data_ptr() for t in qw]])
        setattr(config, name + '_scales', [[t.data_ptr() for t in scales]])
    return ext, torch, cpu, config, mask, qw, scales


@pytest.mark.parametrize('mode', ['fp16', 'int8'])
@pytest.mark.parametrize('contiguous', [False, True])
def test_zero_weights_and_incremental_output(setup, monkeypatch, mode, contiguous):
    ext, torch, cpu, config, mask, qw, scales = setup
    monkeypatch.setenv('KT_SYCL_XMX_MODE', mode)
    if contiguous:
        for name in ('gate', 'up', 'down'):
            setattr(config, name + '_projs', [])
            setattr(config, name + '_scales', [])
            setattr(config, name + '_proj', qw.data_ptr())
            setattr(config, name + '_scale', scales.data_ptr())
    moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    cpu.submit(moe.load_weights_task()); cpu.sync()
    for rows in (1, 17):
        qlen = torch.tensor([rows], dtype=torch.int32)
        x = torch.zeros(rows, 128, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1]] * rows, dtype=torch.int64)
        weights = torch.full((rows, 2), 0.5)
        out = torch.ones_like(x)
        cpu.submit(moe.forward_task(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), False)); cpu.sync()
        assert not out.count_nonzero()
        out.fill_(1)
        cpu.submit(moe.forward_task(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), out.data_ptr(), True)); cpu.sync()
        assert bool((out == 1).all())


def test_invalid_quantization_is_rejected(setup):
    ext, _, _, config, *_ = setup
    config.quant_config.zero_point = True
    with pytest.raises(RuntimeError, match='symmetric GPTQ INT4'):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_invalid_mode_is_rejected(setup, monkeypatch):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv('KT_SYCL_XMX_MODE', 'typo')
    with pytest.raises(RuntimeError, match='KT_SYCL_XMX_MODE'):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


def test_invalid_mapping_is_rejected(setup):
    ext, torch, _, config, *_ = setup
    mapping = torch.tensor([0, 1, 2, 1 << 40], dtype=torch.int64)
    config.physical_to_logical_map = mapping.data_ptr()
    moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    with pytest.raises(RuntimeError, match='physical_to_logical_map'):
        moe.load_weights()


@pytest.mark.parametrize('numa', ['0', '1'])
def test_workspace_bounds_are_rejected(setup, monkeypatch, numa):
    ext, torch, cpu, config, *_ = setup
    monkeypatch.setenv('KT_SYCL_XMX_NUMA_DISPATCH', numa)
    moe = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    moe.load_weights()
    x = torch.zeros(18, 128, dtype=torch.bfloat16)
    ids = torch.zeros(18, 2, dtype=torch.int64)
    weights = torch.ones(18, 2)
    qlen = torch.tensor([18], dtype=torch.int32)
    with pytest.raises(RuntimeError, match='workspace'):
        moe.forward(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), x.data_ptr(), False)
    cpu.submit(moe.forward_task(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), x.data_ptr(), False))
    with pytest.raises(RuntimeError, match='workspace'):
        cpu.sync()
    # The queue remains usable after reporting an invalid request.
    qlen[0] = 1
    cpu.submit(moe.forward_task(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(), x.data_ptr(), x.data_ptr(), False))
    cpu.sync()


@pytest.mark.parametrize('name,value', [('KT_SYCL_XMX_ROWS', '16'), ('KT_SYCL_XMX_GRAPH', 'yes'),
                                        ('KT_SYCL_XMX_HOST_QUANT', 'auto'), ('KT_SYCL_XMX_FUSED_QUANT', 'yes'),
                                        ('KT_SYCL_XMX_NUMA_DISPATCH', 'auto')])
def test_invalid_tuning_is_rejected(setup, monkeypatch, name, value):
    ext, _, _, config, *_ = setup
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        ext.moe.SYCLXMXGPTQInt4_MOE(config)


@pytest.mark.parametrize('fused', ['0', '1', 'prefill'])
def test_graph_reads_changing_inputs_and_routes(setup, monkeypatch, fused):
    ext, torch, cpu, config, mask, qw, scales = setup
    generator = torch.Generator().manual_seed(20261014)
    qw.copy_(torch.randint(-(1 << 31), (1 << 31) - 1, qw.shape, dtype=torch.int32, generator=generator))
    scales.fill_(0.003)
    monkeypatch.setenv('KT_SYCL_XMX_FUSED_QUANT', fused)
    monkeypatch.setenv('KT_SYCL_XMX_GRAPH', '0')
    direct = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    monkeypatch.setenv('KT_SYCL_XMX_GRAPH', '1')
    graph = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    for moe in (direct, graph):
        cpu.submit(moe.load_weights_task())
        cpu.sync()
    qlen = torch.tensor([1], dtype=torch.int32)
    for iteration in range(12):
        topk = iteration % 2 + 1
        x = torch.randn(1, 128, generator=generator).bfloat16()
        ids = torch.randperm(4, generator=generator)[:topk].reshape(1, topk)
        if iteration % 3 == 0:
            ids[0, 0] = -1
        weights = torch.softmax(torch.randn(1, topk, generator=generator), 1)
        outputs = []
        for moe in (direct, graph):
            out = torch.empty_like(x)
            cpu.submit(moe.forward_task(qlen.data_ptr(), topk, ids.data_ptr(), weights.data_ptr(),
                                       x.data_ptr(), out.data_ptr(), False))
            cpu.sync()
            outputs.append(out)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize('rows', [1, 17])
@pytest.mark.parametrize('fused_mode', ['1', 'prefill'])
def test_fused_quantization_matches_unfused(setup, monkeypatch, rows, fused_mode):
    ext, torch, cpu, config, mask, qw, scales = setup
    generator = torch.Generator().manual_seed(42)
    qw.copy_(torch.randint(-(1 << 31), (1 << 31) - 1, qw.shape, dtype=torch.int32, generator=generator))
    scales.fill_(0.003)
    monkeypatch.setenv('KT_SYCL_XMX_FUSED_QUANT', '0')
    direct = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    monkeypatch.setenv('KT_SYCL_XMX_FUSED_QUANT', fused_mode)
    fused = ext.moe.SYCLXMXGPTQInt4_MOE(config)
    qlen = torch.tensor([rows], dtype=torch.int32)
    x = torch.randn(rows, 128, generator=generator).bfloat16()
    ids = torch.tensor([[0, 1]] * rows, dtype=torch.int64)
    weights = torch.full((rows, 2), 0.5)
    outputs = []
    for moe in (direct, fused):
        cpu.submit(moe.load_weights_task())
        cpu.sync()
        out = torch.empty_like(x)
        cpu.submit(moe.forward_task(qlen.data_ptr(), 2, ids.data_ptr(), weights.data_ptr(),
                                   x.data_ptr(), out.data_ptr(), False))
        cpu.sync()
        outputs.append(out)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
