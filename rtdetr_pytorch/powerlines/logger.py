import os
import re
from typing import Optional

import neptune
from neptune import Run


def create_neptune_run(name: str, resume: bool = False, from_run_id: Optional[int] = None) -> Run:
    # Attach to the existing run if loading from checkpoint and resuming the run
    with_id = f"POW-{from_run_id}" if from_run_id is not None and resume else None
    run_name = name if with_id is None else None

    return neptune.init_run(
        project="jakubg/powerlines",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        with_id=with_id,
        name=run_name,
        mode="debug",  # FIXME: change to async
        capture_stdout=True,
        capture_stderr=True,
        capture_traceback=True,
        capture_hardware_metrics=True,
        flush_period=300,
        source_files=[]  # do not log source code
    )


neptune_id_pattern = re.compile(r"\w+-(?P<id>\d+)")


def run_id(run: neptune.Run) -> int:
    neptune_id = run["sys/id"].fetch()
    match = neptune_id_pattern.match(neptune_id)
    return int(match.group("id"))
