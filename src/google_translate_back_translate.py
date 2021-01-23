from googletrans import Translator, LANGUAGES
from typing import Union, List
import numpy as np

translator = Translator()


def backtranslate(
    text: Union[List[str], str],
    n_langs: int = 1,
    random_state: int = 42,
    filter_out_identical=True,
):
    langs = np.random.choice(list(LANGUAGES.keys()), size=n_langs, replace=False)

    if isinstance(text, str):
        ret = []
        for lang in langs:
            translated = translator.translate(text, dest=lang)
            ret.append(translator.translate(translated.text, src=lang, dest="en").text)
    else:  # Then we have a list of strings
        ret = []

        for lang in langs:
            to_append = []
            for t in text:
                translated = translator.translate(t, dest=lang)
                to_append.append(
                    translator.translate(translated.text, src=lang, dest="en").text
                )
        ret.append(to_append)

    return ret, langs


if __name__ == "__main__":
    s, langs = backtranslate(
        "Do we drive on the right side of the road in Sweden?", n_langs=20
    )
    print(set(s))
