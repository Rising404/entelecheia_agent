"""隔离复现 MPS 设备探测竞争；显式运行，不进入普通 pytest 或调用远端 API。

直接运行生产 select_device；修复后的双线程必须仅靠生产设备锁通过，诊断脚本
不再额外套锁。历史修复前的 locked_parallel 对照保留在运行记录中。
所有日志仅记录调用栈、设备和测试序号。
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


MODES = ("single", "parallel")
REPO_ROOT = Path(__file__).resolve().parents[3]


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def child(mode: str, iterations: int) -> int:
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.enable(all_threads=True)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import torch

    from personagraph.retrieval.compute import devices

    emit(
        "environment",
        pid=os.getpid(),
        torch=torch.__version__,
        mode=mode,
        mps_available=torch.backends.mps.is_available(),
        iterations=iterations,
    )
    if not torch.backends.mps.is_available():
        return 77
    # 所有组都先在主线程完成同样的初始化，排除首次导入/设备初始化差异。
    assert devices.select_device("mps").device == "mps"
    original_probe = devices._probe
    seen_threads: set[int] = set()
    log_lock = threading.Lock()

    def traced_probe(module: object, device: str) -> None:
        thread = threading.current_thread()
        with log_lock:
            if thread.ident not in seen_threads:
                seen_threads.add(thread.ident)
                emit(
                    "probe_caller",
                    thread=thread.name,
                    native_id=thread.native_id,
                    stack=traceback.format_stack(),
                )
        original_probe(module, device)

    devices._probe = traced_probe
    barrier = threading.Barrier(2) if mode != "single" else None
    failures: list[str] = []
    completed: dict[str, int] = {}

    def loop() -> None:
        try:
            for ordinal in range(iterations * (2 if mode == "single" else 1)):
                if barrier is not None:
                    barrier.wait(timeout=10)
                emit(
                    "probe_begin",
                    thread=threading.current_thread().name,
                    ordinal=ordinal,
                )
                assert devices.select_device("mps").device == "mps"
                emit(
                    "probe_end", thread=threading.current_thread().name, ordinal=ordinal
                )
                completed[threading.current_thread().name] = ordinal + 1
        except BaseException:
            failures.append(traceback.format_exc())
            if barrier is not None:
                barrier.abort()

    worker = None
    if barrier is not None:
        worker = threading.Thread(target=loop, name="post-commit-probe")
        worker.start()
    loop()
    if worker is not None:
        worker.join(timeout=15)
        if worker.is_alive():
            faulthandler.dump_traceback(all_threads=True)
            raise RuntimeError("diagnostic thread did not finish")
    emit("child_complete", failures=failures, completed=completed)
    return int(bool(failures))


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONFAULTHANDLER="1"
    )
    records = []
    for mode in args.modes:
        for trial in range(args.repeats):
            name = f"{mode}-{trial + 1}"
            stdout_path = output / f"{name}.stdout.log"
            stderr_path = output / f"{name}.stderr.log"
            argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                mode,
                "--iterations",
                str(args.iterations),
                "--allow-local-gpu",
            ]
            started = time.monotonic()
            timed_out = False
            with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
                process = subprocess.Popen(
                    argv, env=environment, stdout=stdout, stderr=stderr, cwd=REPO_ROOT
                )
                try:
                    process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    process.wait(timeout=10)
            record = {
                "mode": mode,
                "trial": trial + 1,
                "pid": process.pid,
                "exit_code": process.returncode,
                "timed_out": timed_out,
                "elapsed_s": round(time.monotonic() - started, 3),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            records.append(record)
            emit("trial_complete", **record)
            # 保存每次已完成结果；后续诊断被停止时不丢掉已收集的证据。
            with (output / "results.json").open("w") as stream:
                json.dump(records, stream, ensure_ascii=False, indent=2)
            if process.returncode == 77:
                return 77
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-local-gpu", action="store_true")
    parser.add_argument("--child", choices=MODES)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.allow_local_gpu:
        parser.error("requires explicit --allow-local-gpu")
    if not 1 <= args.iterations <= 1000 or not 1 <= args.repeats <= 10:
        parser.error("iterations must be 1..1000 and repeats 1..10")
    if not 1 <= args.timeout <= 60:
        parser.error("timeout must be 1..60 seconds per child")
    if args.child:
        return child(args.child, args.iterations)
    if args.output is None:
        parser.error("parent requires a new --output directory")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
