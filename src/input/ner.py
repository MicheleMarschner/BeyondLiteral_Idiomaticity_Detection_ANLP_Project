import torch
from transformers import pipeline


# load NER models for each language, using GPU if available
device = 0 if torch.cuda.is_available() else -1

ner_models = {
    "EN": pipeline(
        "ner",
        model="Babelscape/wikineural-multilingual-ner",
        aggregation_strategy="max",
        device=device,
    ),
    "PT": pipeline(
        "ner",
        model="Babelscape/wikineural-multilingual-ner",
        aggregation_strategy="max",
        device=device,
    ),
    "GL": pipeline(
        "ner",
        model="marcosgg/bert-base-gl-SLI-NER",
        aggregation_strategy="max",
        device=device,
    ),
}



'''
def apply_ner(text: str, language: str) -> str:
    """
    Inline NER tagging of a text.
    
    Returns:
        Text with inline entity tags.
    """

    if not text or language not in ner_models:
        return text

    model = ner_models[language]
    entities = model(text)

    # Sort by descending offset to prevent index shift
    entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)

    tagged_text = text

    for ent in entities_sorted:
        start, end = ent["start"], ent["end"]
        label = ent["entity_group"]

        tagged_text = (
            tagged_text[:start]
            + f"[{label}] {tagged_text[start:end]} [/{label}]"
            + tagged_text[end:]
        )

    return tagged_text

'''

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

        model = ner_models[lang]

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