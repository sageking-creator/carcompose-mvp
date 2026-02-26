from typing import Any, Dict

from settings import Settings
from exceptions import HarmonyScoreTooLowError, InvalidInputError


def _error(message: str, settings: Settings) -> Dict[str, Any]:
    return {
        "status": "error",
        "message": message,
        "workerBuildId": settings.worker_build_id,
    }


def run_composite(payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    try:
        from pipeline import run_pipeline

        return run_pipeline(payload, settings)
    except HarmonyScoreTooLowError as error:
        return {
            "status": "rejected",
            "variant": "full",
            "workerBuildId": settings.worker_build_id,
            "score": error.score,
            "guidance": error.guidance,
        }
    except InvalidInputError as error:
        return _error(str(error), settings)
    except Exception as error:
        return _error(str(error), settings)
