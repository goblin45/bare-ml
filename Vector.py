import random
from typing import List, Optional, Union, Tuple


class Vector:
    def __init__(self, v: Optional[Union[List[float], List[int], None]] = None):
        if v is None:
            self.values = []
        else:
            self.values = v

    def push_front(self, p: float) -> None:
        self.values = [p] + self.__get__()

    def push_back(self, p: float) -> None:
        self.values = self.__get__() + [p]

    def generate(self, size: int, g_range: Tuple[float, float], floating : bool = False) -> None:
        start, end = round(g_range[0]), round(g_range[1])
        self.values = [(random.randint(start, end) if not floating else random.uniform(start, end) * 0.1) for _ in range(size)]

    def replace(self, index: int, value: float) -> None:
        if index < 0:
            raise ValueError("Index cannot be negative")
        if index >= self.__len__():
            raise ValueError("Index out of range")
        self.values[index] = value

    def dot(self, v: 'Vector') -> float:
        if v is None or self.__len__() != v.__len__():
            raise ValueError("Vectors must have same length to get dot product")
        total : float = 0.0
        for i in range(self.__len__()):
            total += self.__getitem__(i) * v.__getitem__(i)
        return total

    def __print__(self):
        print(self.values)

    def __get__(self):
        return self.values
    
    def __len__(self):
        return len(self.values)
    
    def __iter__(self):
        return iter(self.values)
    
    def __getitem__(self, index: int) -> float:
        return self.values[index]