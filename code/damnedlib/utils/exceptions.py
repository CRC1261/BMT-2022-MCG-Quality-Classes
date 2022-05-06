import numpy as np
import damnedlib.utils.checks as checks


class ArrayShapeMismatchError(Exception):
    def __init__(self, array_1: np.ndarray, array_2: np.ndarray) -> None:
        checks.check_type(array_1, np.ndarray)
        checks.check_type(array_2, np.ndarray)
        super().__init__(
            f"Shapes of arrays should be equal but are: {array_1.shape} and {array_2.shape}."
        )


class WrongTypeError(TypeError):
    def __init__(actual: type, *args: type) -> None:
        if len(args) == 0:
            raise ValueError("Expected at least one type but received none.")
        expected = str(args[0])
        for i in range(len(args) - 2):
            expected += ", " + str(args[i])
        if len(args) > 1:
            expected += " or " + str(args[-1])

        super().__init__(f"Received type {actual} is not {expected}.")
