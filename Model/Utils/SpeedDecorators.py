import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]


import time

def monitor_speed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the actual function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        print(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def monitor_culm_speed(func):
    culm_time = 0  # Initialize cumulative time

    def wrapper(*args, **kwargs):
        nonlocal culm_time  # Use nonlocal instead of global
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the actual function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        culm_time += execution_time  # Update cumulative execution time
        print(f"Function {func.__name__} executed in {culm_time:.4f} seconds")
        return result

    return wrapper