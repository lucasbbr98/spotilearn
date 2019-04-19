INITIAL_MESSAGE_ERROR = 'Debug warning: You have used @console without specifying an initial message argument'


def console(initial_message=INITIAL_MESSAGE_ERROR, final_message=None):
    def decorator(function):
        def wrapper(*args, **kwargs):
            print(initial_message)
            result = function(*args, **kwargs)
            if final_message:
                print(final_message)
            return result
        return wrapper
    return decorator
