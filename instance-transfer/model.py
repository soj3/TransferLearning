from abc import ABC, abstractmethod, abstractproperty
from typing import List, Any, Tuple, Dict
from example import Example


class AbstractModel(ABC):
    """
    Abstract model definition for a binary classifier
    """

    @abstractmethod
    def fit(self, examples: List[Example]) -> None:
        pass

    @abstractmethod
    def classify(self, example: Example) -> Tuple[bool, float]:
        pass
