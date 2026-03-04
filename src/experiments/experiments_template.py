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



EXPERIMENTS = ExperimentTemplate(
    settings=["one_shot"],
    language_mode="per_language",            # "cross_lingual", "multilingual" !TODO: if [] grid needs to be adapted
    languages=["EN"],
    input_variant=[
        {"context": "previous_target_next", "include_mwe_segment": True, "transform": "none", "features": []}
    ],
    model_families=["mBERT"],
    seeds=[51]
)