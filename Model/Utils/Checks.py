import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]


from functools import wraps
import inspect
from typing import Optional, Union, get_origin, get_args
from types import FunctionType
from abc import ABC, ABCMeta


class Check:
    @staticmethod
    def type_check_arguments(func):
        signature = inspect.signature(func)
        type_hints = {
            name: param.annotation
            for name, param in signature.parameters.items()
            if param.annotation is not inspect._empty
        }

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            
            for arg_name, arg_value in bound_arguments.arguments.items():
                if arg_name in type_hints:
                    expected_type = type_hints[arg_name]
                    origin = get_origin(expected_type)
                    args = get_args(expected_type)
                    
                    # Handle Union types
                    if origin is Union:
                        if not any(isinstance(arg_value, t) for t in args):
                            raise TypeError(f"Argument '{arg_name}' must be one of {args}, got {type(arg_value)}.")
                    
                    # Handle Optional (which is Union[T, None])
                    elif origin is Union and type(None) in args:
                        allowed = tuple(t for t in args if t is not type(None))
                        if not isinstance(arg_value, allowed + (type(None),)):
                            raise TypeError(f"Argument '{arg_name}' must be {allowed} or None, got {type(arg_value)}.")
                    
                    # Handle other types
                    else:
                        if not isinstance(arg_value, expected_type):
                            raise TypeError(f"Argument '{arg_name}' must be {expected_type}, got {type(arg_value)}.")
            return func(*args, **kwargs)
        return wrapper

class AutoTypeCheckMeta(type):
    def __new__(cls, name, bases, namespace):
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, FunctionType):
                namespace[attr_name] = Check.type_check_arguments(attr_value)
        return super().__new__(cls, name, bases, namespace)
from functools import wraps
import inspect
from typing import Optional, Union
from types import FunctionType

class Check:
    @staticmethod
    def type_check_arguments(func):
        """
        A decorator to type-check the arguments of a function based on its type hints.
        Raises a TypeError if any argument does not match the specified type.
        """
        # Extract function signature and type hints once, at decoration time
        signature = inspect.signature(func)
        type_hints = {
            name: param.annotation
            for name, param in signature.parameters.items()
            if param.annotation is not inspect._empty
        }

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to their names
            try:
                bound_arguments = signature.bind(*args, **kwargs)
                bound_arguments.apply_defaults()  # Include default arguments
            except TypeError as e:
                raise TypeError(e)
            
            # Check argument types
            for arg_name, arg_value in bound_arguments.arguments.items():
                if arg_name in type_hints and not isinstance(arg_value, type_hints[arg_name]):
                    raise TypeError(
                        f"Argument '{arg_name}' must be of type {type_hints[arg_name].__name__}, "
                        f"but got {type(arg_value).__name__}."
                    )
            return func(*args, **kwargs)

        return wrapper
    

# Metaclass to auto-apply type checking
class AutoTypeCheckMeta(ABCMeta):
    def __new__(cls, name, bases, namespace):
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, FunctionType):
                # Apply the type-check decorator to all methods
                namespace[attr_name] = Check.type_check_arguments(attr_value)
        return super().__new__(cls, name, bases, namespace)


class TypeCheckingClass(ABC, metaclass=AutoTypeCheckMeta):
    pass
        
if __name__ == "__main__":
    
    # type_check_arguments
    @Check.type_check_arguments
    def test(a:str):...
    test('')
    
    @Check.type_check_arguments
    def test(a: Optional[str]):...
    test('')
    test(None)
    
    @Check.type_check_arguments
    def test(a: Union[str,int]):...
    test('')
    test(1)
    
    
    # Class Based argument checks
    class testClass(TypeCheckingClass):
        def test(s: str):...
    testClass.test('')
    
            
            
        
    
        

if __name__ == "__main__":

    # No Type Check
    class TestClass(TypeCheckingClass):
        def test_method(self, s):
            pass
        
    obj = TestClass()
    obj.test_method("valid")  # Valid
        
    # Optional Type Check
    class TestClass(TypeCheckingClass):
        def test_method(self, s: Optional[str]):
            pass
        
    obj = TestClass()
    obj.test_method("valid")  # Valid
    obj.test_method(None)  # Valid
    
    # Multiple Type Check
    class TestClass(TypeCheckingClass):
        def test_method(self, s: Union[str, int]):
            pass

    obj = TestClass()
    obj.test_method("valid")  # Valid
    obj.test_method(1)  # Valid
    