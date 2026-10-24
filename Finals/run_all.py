"""Run each lab in a separate process and retain its console output."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    output = folder / "outputs"
    output.mkdir(exist_ok=True)
    flags = [flag for flag in ("--quick", "--no-show")
             if getattr(args, flag[2:].replace("-", "_"))]
    results = []
    for script in sorted(folder.glob("*.py")):
        if script.name.startswith("_") or script.name == Path(__file__).name:
            continue
        print(f"Running {script.name}...", flush=True)
        start = time.monotonic()
        log = output / f"{script.stem}.log"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                [sys.executable, "-u", str(script), *flags],
                stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        result = {"script": script.name, "exit_code": process.returncode,
                  "seconds": round(time.monotonic() - start, 1)}
        results.append(result)
        print(f"{'PASS' if process.returncode == 0 else 'FAIL'}: {result}", flush=True)
    (output / "summary.json").write_text(
        json.dumps({"quick": args.quick, "results": results}, indent=2),
        encoding="utf-8",
    )
    return int(any(result["exit_code"] for result in results))


if __name__ == "__main__":
    sys.exit(main())
