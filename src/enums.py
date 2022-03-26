from enum import Enum

class TrainingType(Enum):
    TWO_DATASETS_SEQUENTIALLY = 1
    TWO_DATASETS_SIMULTANEOUSLY = 2
    ONE_DATASET = 3