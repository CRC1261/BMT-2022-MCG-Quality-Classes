from damnedlib.utils.exceptions import WrongTypeError


def check_type(variable: any, *args: type) -> None:
    """
    Checks if the variable is one of the specified
    types and raises an WrongTypeError if this
    is not the case

    Args:
        variable (any): The Variable to check
        *args (type): The expected types to check against
    """
    if len(args) == 0:
        raise ValueError("Expected at least one type but received none.")
    if isinstance(variable, args):
        return
    raise WrongTypeError(type(variable), args)
