"""Answer generators (all our own, no pre-trained language model):

composer   (default) - question-type classifier + trained sentence ranker + answer composition (ml/composer.py)
extractive (baseline) - sentences with the highest overlap / similarity (ml/generator.py)
"""
from ml.composer import Composer  # noqa: F401
from ml.generator import ExtractiveGenerator  # noqa: F401

AVAILABLE_GENERATORS = ("composer", "extractive")
