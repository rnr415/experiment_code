from abc import ABC, abstractmethod
from typing import List, Callable

class BaseAttribute(ABC):
    def __init__(self, value, functions: List[Callable] = None):
        self.value = value
        self.functions = functions or []

    def add_function(self, function: Callable):
        self.functions.append(function)

    def pre_process(self):
        result = self.value
        for proc_function in self.functions:
            result = proc_function(result)
        return result

class TextAttribute(BaseAttribute):
    def __init__(self, value, text_functions: List[Callable] = None):
        super().__init__(value, text_functions)

class NumberAttribute(BaseAttribute):
    def __init__(self, value, number_functions: List[Callable] = None):
        super().__init__(value, (number_functions or []))

class AlphanumericAttribute(BaseAttribute):
    def __init__(self, value, alphanumeric_functions: List[Callable] = None):
        super().__init__(value, (alphanumeric_functions or []))

# Example usage
text_functions = [lambda x: x.strip(), lambda x: x.lower(), lambda x: x.replace('o', '0')]
text_attr = TextAttribute("  Hello World  ", text_functions)

number_functions = [lambda x: float(x), lambda x: x * 2 ]
number_attr = NumberAttribute("42", number_functions)

alphanumeric_functions = [lambda x: ''.join(char for char in x if char.isalnum())] 
alphanumeric_attr = AlphanumericAttribute("A1b2C3!", alphanumeric_functions)

print(text_attr.pre_process())  # Output: hello world
print(number_attr.pre_process())  # Output: 42.0
print(alphanumeric_attr.pre_process())  # Output: A1b2C3

# Adding custom functions
print(text_attr.pre_process())  # Output: hell0 w0rld

# Using custom functions during initialization
custom_text_attr = TextAttribute("  CUSTOM TEXT  ", [lambda x: x.replace('T', 't')])
print(custom_text_attr.pre_process())  # Output: custom text
