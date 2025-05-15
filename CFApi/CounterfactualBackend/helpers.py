import functools
import warnings
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def deprecated(msg=None):
    
    if msg ==None:
        msg = ""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(f"The function or class '{func.__name__}' is deprecated. Usage is strongly discouraged.\n{msg}", DeprecationWarning, stacklevel=2)
            print(f"{bcolors.FAIL}'{func.__name__}' might get removed at any time!{bcolors.ENDC}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def measure_function_time(fn,times,*args):
    import time
    time_measured = []
    for i in range(times):
        start = time.time()
        fn(*args)
        end = time.time()
        time_measured.append(end-start)
    return np.average(time_measured)
