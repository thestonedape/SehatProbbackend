"""Fail closed before a 512 MB deployment; anonymous RSS is not the gate."""
import json
import sys
from pathlib import Path

def validate(report):
    peak = report.get("peak_cgroup_memory_bytes")
    outcomes = report.get("outcomes", {})
    successes = report.get("status_counts", {}).get("200", outcomes.get("completed", 0))
    return (report.get("memory_limit") == "512m"
            and report.get("requests", 0) >= 100
            and successes == report["requests"]
            and isinstance(peak, int) and 0 < peak < 400_000_000
            and report.get("oom_kill_events") == 0
            and report.get("container_oom_killed") is False)

if __name__ == "__main__":
    report = json.loads(Path(sys.argv[1]).read_text())
    if not validate(report):
        raise SystemExit("Deployment blocked: require >=100 completed requests, total cgroup peak <400 MB, and zero OOM events.")
    print("512 MB capacity gate passed; verify benchmark revision and live configuration before rollout.")
