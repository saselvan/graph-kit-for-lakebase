"""The serverless environment every kit job runs in, and the runtime record a notebook run writes.

Serverless picks the job's Python from its environment version, not from the wheel. A notebook
task with no environment runs the default, environment version 1 = Python 3.10.12 (live,
2026-10-04: `client.1.13`), and the kit needs Python >= 3.12 (pyproject requires-python). The
docs list the Python of each version (https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/):
1 = 3.10.12, 2 = 3.11.10, 3 to 6 = 3.12.3. So every kit job carries a job environment pinned
to ENVIRONMENT_VERSION, and every notebook task names it by environment_key (a notebook task
honours a job-level environment: live, environment_key + version 3 ran Python 3.12.3,
`client.3.6`; docs https://docs.databricks.com/aws/en/compute/serverless/dependencies).
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ENVIRONMENT_KEY = "lgk"
ENVIRONMENT_VERSION = "3"  # Python 3.12.3 (docs, above); 1 = 3.10 and 2 = 3.11 cannot run the kit
RUNTIME_FILE = "runtime.json"


def job_environment(dependencies: Sequence[str] = ()) -> Any:
    """The SDK JobEnvironment every kit job carries (key ENVIRONMENT_KEY, pinned version)."""
    from databricks.sdk.service import compute, jobs

    return jobs.JobEnvironment(
        environment_key=ENVIRONMENT_KEY,
        spec=compute.Environment(environment_version=ENVIRONMENT_VERSION,
                                 dependencies=list(dependencies) or None),
    )


def runtime_record(environment_version_requested: Optional[str], steps: List[str]) -> Dict[str, Any]:
    """What this process runs on: Python, and the serverless environment it was given."""
    return {
        "python_version": sys.version,
        "python_version_info": list(sys.version_info[:3]),
        # Serverless sets DATABRICKS_RUNTIME_VERSION to client.<environment version>.<patch>.
        "environment_version": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        "environment_version_requested": environment_version_requested or None,
        "steps": list(steps),
    }


def write_runtime_record(run_folder: Path, environment_version_requested: Optional[str],
                         steps: List[str]) -> Path:
    """Write `<run_folder>/runtime.json` (runner-specific: parity skips it)."""
    path = Path(run_folder) / RUNTIME_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_record(environment_version_requested, steps), indent=2))
    return path
