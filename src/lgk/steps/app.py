"""App step: generate the search app folder and print its one deploy command.

The step never deploys: it prints `databricks apps deploy ...` for the operator to run with
their own permissions, and calls no Apps API.
"""

from typing import Any, Dict

from lgk.app.generate import generate_app


def app(config: Dict[str, Any]) -> Dict[str, str]:
    """Write the app folder (default `<run_folder>/app/<name>`) and print the deploy command."""
    result = generate_app(config)
    print(result["deploy_command"])
    return result
