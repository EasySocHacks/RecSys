class RecModelException(Exception):
    def __init__(
        self,
        error_message: str,
    ) -> None:
        self.error_message = error_message
        super().__init__(error_message)


class RecModelNotLearnedYetException(RecModelException):
    def __init__(
        self,
        model_name: str,
        error_message: str = "Model '{}' haven't been learned yet"
    ):
        super().__init__(error_message.format(model_name))
