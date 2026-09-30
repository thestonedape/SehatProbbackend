import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("capacity", Path(__file__).resolve().parents[1] / "scripts/check_capacity.py")
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)

def test_total_container_memory_required():
    report = {"memory_limit":"512m","requests":100,"status_counts":{"200":100},"peak_cgroup_memory_bytes":390_000_000,"peak_anon_memory_bytes_sampled":350_000_000,"oom_kill_events":0,"container_oom_killed":False}
    assert capacity.validate(report)
    report["peak_cgroup_memory_bytes"] = 512 * 1024 * 1024
    assert not capacity.validate(report)
    report["peak_cgroup_memory_bytes"] = 390_000_000
    report["oom_kill_events"] = 1
    assert not capacity.validate(report)
    del report["peak_cgroup_memory_bytes"]
    assert not capacity.validate(report)
