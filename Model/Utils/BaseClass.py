# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Utils.Checks import AutoTypeCheckMeta, Union, Optional, Check, TypeCheckingClass
from abc import ABC, abstractmethod, ABCMeta

class BaseClass(ABC, metaclass=AutoTypeCheckMeta):...

