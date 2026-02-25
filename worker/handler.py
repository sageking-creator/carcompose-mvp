from typing import Any, Dict

import runpod

from actions.composite import run_composite
from actions.download_models import run_download_models
from settings import get_settings


def _error(message: str, status: str = "error") -> Dict[str, Any]:
    return {"status": status, "message": message}


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    payload = job.get("input", {})
    action = payload.get("action")

    # Important: let init job failures bubble up so RunPod marks the job FAILED.
    # `/api/ready` relies on RunPod job status for readiness.
    if action == "download_models":
        result = run_download_models(get_settings())
        return {"status": "success", **result}

    try:
        if action == "composite":
            return run_composite(payload, get_settings())

        return _error("Unsupported action. Use 'download_models' or 'composite'.")
    except Exception as error:
        return _error(str(error))


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
