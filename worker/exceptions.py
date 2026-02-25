class CarComposeError(Exception):
    pass


class InvalidInputError(CarComposeError):
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Invalid {field}: {reason}")


class ModelInferenceError(CarComposeError):
    def __init__(self, model: str, original: Exception):
        self.model = model
        self.original = original
        super().__init__(f"{model} failed: {original}")


class ControlComSetupError(CarComposeError):
    pass


class HarmonyScoreTooLowError(CarComposeError):
    def __init__(self, score: float, guidance: list[str]):
        self.score = score
        self.guidance = guidance
        super().__init__(f"Harmony score too low: {score:.4f}")

