from enum import Enum

class TrainingType(Enum):
    TWO_DATASETS_SEQUENTIALLY = 1
    TWO_DATASETS_SIMULTANEOUSLY = 2
    ONE_DATASET = 3


class Language(Enum):
    syriac = 1
    hebrew = 2

class WordGrammarVersion(Enum):
    SSI = 1
    synvar = 2

class APICodes(Enum):
    CORRECT = 1
    NOTCHECKED = 2
    NOTCORRECT = 3
