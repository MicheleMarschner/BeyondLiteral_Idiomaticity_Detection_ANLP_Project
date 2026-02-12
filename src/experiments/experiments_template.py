from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class ExperimentTemplate:
    """Defines the experiment grid used to generate runs."""
    settings: List[str] 
    language_mode: str
    languages: List[str]
    input_variant: List[Dict[str, Any]]
    model_families: List[str]
    seeds: List[int]
'''
EXPERIMENTS = ExperimentTemplate(
    settings=["zero_shot", "one_shot"],
    language_mode="per_language",
    languages=["EN"],
    input_variant=[
        {"context": "target", "include_mwe_segment": False, "transform": "none", "features": []},
        {"context": "prev_target_next", "include_mwe_segment": False, "transform": "none", "features": ["ner"]},
    ],
    model_families=["logreg_tfidf", "logreg_word2vec"],
    seeds=[51]
)

'''
EXPERIMENTS = ExperimentTemplate(
    settings=["zero_shot"],
    language_mode="per_language",
    languages=["EN"],
    input_variant=[
        {"context": "target", "include_mwe_segment": False, "transform": "none", "features": []},
    ],
    model_families=["logreg_tfidf"],
    seeds=[51]
)