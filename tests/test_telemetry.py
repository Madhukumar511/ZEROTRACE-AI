import pytest
from zerotrace.telemetry.hardware import (
    get_cpu_power_estimate,
    get_gpu_telemetry,
    read_rapl_energy_uj,
)
from zerotrace.telemetry.process import get_isolated_ai_cpu
from zerotrace.telemetry.aggregator import get_total_system_power
from zerotrace.core.emissions import get_carbon, set_carbon


def test_cpu_power_estimate_values():
    assert get_cpu_power_estimate(0.0) == 0.0
    assert get_cpu_power_estimate(100.0) >= 35.0
    half_power = get_cpu_power_estimate(50.0)
    full_power = get_cpu_power_estimate(100.0)
    assert half_power == pytest.approx(full_power / 2.0, rel=0.1)


def test_gpu_telemetry_structure():
    gpu = get_gpu_telemetry()
    assert "gpu_util_pct" in gpu
    assert "gpu_power_w" in gpu
    assert "available" in gpu
    assert isinstance(gpu["gpu_power_w"], (int, float))


def test_system_power_aggregation_structure():
    agg = get_total_system_power(30.0)
    assert "cpu_w" in agg
    assert "gpu_w" in agg
    assert "total_w" in agg
    assert agg["total_w"] >= agg["cpu_w"]


def test_isolated_ai_cpu_sweeper():
    cpu = get_isolated_ai_cpu()
    assert isinstance(cpu, float)
    assert cpu >= 0.0


def test_carbon_intensity_getter_and_setter():
    orig = get_carbon()
    assert orig > 0
    set_carbon(185.5)
    assert get_carbon() == 185.5
    set_carbon(orig)  # restore


def test_rapl_reader_safe_execution():
    # Will either return int or None without throwing exception
    rapl_val = read_rapl_energy_uj()
    assert rapl_val is None or isinstance(rapl_val, int)
