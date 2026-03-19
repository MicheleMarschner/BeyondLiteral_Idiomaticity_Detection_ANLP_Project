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
    settings=["zero_shot"],
    language_mode="per_language",
    languages=["PT"],
    input_variant=[
        {"context": "previous_target_next", "include_mwe_segment": True, "transform": "none", "features": []},
        #{"context": "previous_target_next", "include_mwe_segment": True, "transform": "none", "features": ["glosses"]},
        #{"context": "previous_target_next", "include_mwe_segment": True, "transform": "none", "features": ["ner"]},
        #{"context": "previous_target_next", "include_mwe_segment": True, "transform": "highlight", "features": []},
        #{"context": "previous_target_next", "include_mwe_segment": True, "transform": "highlight", "features": ["glosses"]},
        #{"context": "previous_target_next", "include_mwe_segment": True, "transform": "highlight", "features": ["ner"]},

        #{"context": "target", "include_mwe_segment": True, "transform": "none", "features": []},
        #{"context": "target", "include_mwe_segment": True, "transform": "none", "features": ["glosses"]},
        #{"context": "target", "include_mwe_segment": True, "transform": "none", "features": ["ner"]},
        #{"context": "target", "include_mwe_segment": True, "transform": "highlight", "features": []},
        #{"context": "target", "include_mwe_segment": True, "transform": "highlight", "features": ["glosses"]},
        #{"context": "target", "include_mwe_segment": True, "transform": "highlight", "features": ["ner"]},

    ],
    #model_families=["logreg_tfidf", "logreg_word2vec", "mBERT"],
    model_families=["mBERT"],
    seeds=[51]
)