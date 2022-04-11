from scipy.sparse import data

from .base import Base_Evaluator
from .epic_kitchens100 import EPIC_KITCHENS100_Evaluator

def build_evaluator(dataset):
    if 'EPIC-KITCHENS-100' in dataset.name:
        return EPIC_KITCHENS100_Evaluator(dataset)
    else:
        return Base_Evaluator(dataset)