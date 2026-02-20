import nltk
from nltk.corpus import wordnet as wn
from typing import List

nltk.download('wordnet')
nltk.download('omw-1.4')

# print(wn.langs())

def get_glosses(word, language, num=2)-> List[str]:
    """
    Retrieve up to 'num' glosses (definitions) for a word.
    If there is no glasses, return an empty list.
    output is a list glosses.

    Language codes:
    PT -> por
    EN -> eng
    GL -> glg
    """

    # Map dataset language codes to WordNet codes
    lang_map = {
        "PT": "por",
        "EN": "eng",
        "GL": "glg"
    }

    # Validate language
    if language not in lang_map:
        raise ValueError(
            f"Unsupported language '{language}'. "
            "Allowed values are: ['EN', 'PT', 'GL']."
        )

    wn_lang = lang_map[language]

    word = word.lower()
    synsets = wn.synsets(word, lang = wn_lang)

    if not synsets:
        return []

    # Return up to 'num' definitions safely
    glosses = [syn.definition() for syn in synsets[:num]]

    return glosses