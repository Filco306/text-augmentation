from src.google_translate_back_translate import (
    backtranslate as google_translate_back_translate,
)
from src.MarianMT_backtranslation import backtranslate as MarianMT_backtranslation
from typing import Union, List


METHODS = ["google", "MarianMT"]


def backtranslate(
    text: Union[List[str], str],
    n_variations: int = 4,
    method: str = "google",
    filter_out_identical: bool = True,
    random_state: int = 42,
):
    assert method in METHODS, "Please specify method to be one of {}".format(METHODS)

    if method == "google":
        variations, langs = google_translate_back_translate(
            text, n_langs=n_variations, random_state=random_state
        )

        identical_variations = set()
    elif method == "MarianMT":
        variations, langs = MarianMT_backtranslation(
            text, n_samples=n_variations, random_state=random_state
        )

    if filter_out_identical is True:
        for i, x in enumerate(list(text)):
            if text == x:
                identical_variations.add(i)
    variations = [
        variations[i] for i in range(len(variations)) if i not in identical_variations
    ]
    langs = [langs[i] for i in range(len(langs)) if i not in identical_variations]
    return variations, langs
