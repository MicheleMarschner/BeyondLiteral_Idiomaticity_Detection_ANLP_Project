import torch
from transformers import pipeline

from config import DEVICE, is_cluster_run


# load NER models for each language, using GPU if available

ner_models = {
    "EN": "Babelscape/wikineural-multilingual-ner",
    "PT": "Babelscape/wikineural-multilingual-ner",
    "GL": "marcosgg/bert-base-gl-SLI-NER",
}

_loaded_ner_models = {}

def get_ner_model(lang):
    """Return cached NER pipeline for one language."""
    if lang not in ner_models:
        return None

    if lang not in _loaded_ner_models:
        _loaded_ner_models[lang] = pipeline(
            "ner",
            model=ner_models[lang],
            aggregation_strategy="max",
            device=DEVICE,
            local_files_only=is_cluster_run(),
        )

    return _loaded_ner_models[lang]


# batched version of named_entity_recognition, since row wise application was slow, 
# we apply NER in batch to all texts of the split at once
def apply_ner_batch(texts, languages):
    """
    Inline NER tagging of a text.
    
    Returns:
        Text with inline entity tags.
    """

    device_is_gpu = torch.cuda.is_available()

    # set batch size based on device, larger batch size for GPU, 
    # smaller for CPU to avoid memory issues
    batch_size = 32 if device_is_gpu else 8

    # group indices of texts by language to minimize number of model calls, 
    # since we have one model per language
    grouped_indices = {}
    for idx, lang in enumerate(languages):
        grouped_indices.setdefault(lang, []).append(idx)

    tagged_texts = [None] * len(texts)

    for lang, indices in grouped_indices.items():

        if lang not in ner_models:
            for i in indices:
                tagged_texts[i] = texts[i]
            continue

        model = get_ner_model(lang)

        lang_texts = [texts[i] for i in indices]

        # run batched inference
        outputs = model(lang_texts, batch_size=batch_size)

        for local_idx, entities in enumerate(outputs):

            original_text = lang_texts[local_idx]

            # sort descending offsets to prevent index shift when inserting tags
            entities_sorted = sorted(
                entities,
                key=lambda x: x["start"],
                reverse=True
            )

            tagged = original_text

            for ent in entities_sorted:
                start, end = ent["start"], ent["end"]
                label = ent["entity_group"]

                tagged = (
                    tagged[:start]
                    + f"[{label}] {tagged[start:end]} [/{label}]"
                    + tagged[end:]
                )

            tagged_texts[indices[local_idx]] = tagged

    return tagged_texts