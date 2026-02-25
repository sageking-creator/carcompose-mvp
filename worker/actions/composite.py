from typing import Any, Dict

from settings import Settings
from exceptions import HarmonyScoreTooLowError, InvalidInputError


def _error(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def run_composite(payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    try:
        from pipeline import run_pipeline

        return run_pipeline(payload, settings)
    except HarmonyScoreTooLowError as error:
        return {
            "status": "rejected",
            "variant": "full",
            "score": error.score,
            "guidance": error.guidance,
        }
    except InvalidInputError as error:
        return _error(str(error))
    except Exception as error:
        return _error(str(error))
