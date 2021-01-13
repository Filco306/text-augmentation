from transformers import MarianMTModel, MarianTokenizer
import numpy as np
from typing import Tuple, Union, List
import transformers

MULTILINGUAL_TGTS = ["roa"]


def get_models(
    src: str, tgt: str, verbose: int = 0
) -> Tuple[
    transformers.models.marian.tokenization_marian.MarianTokenizer,
    transformers.models.marian.tokenization_marian.MarianTokenizer,
    transformers.models.marian.modeling_marian.MarianMTModel,
    transformers.models.marian.modeling_marian.MarianMTModel,
]:
    model_to = "Helsinki-NLP/opus-mt-{src}-{tgt}".format(src=SRC_TO, tgt=TGT_TO)
    model_from = "Helsinki-NLP/opus-mt-{src}-{tgt}".format(src=TGT_TO, tgt=SRC_TO)

    if verbose > 0:
        print("Loading models: {} and {}".format(model_to, model_from))
    tokenizer_to = MarianTokenizer.from_pretrained(model_to)
    model_to = MarianMTModel.from_pretrained(model_to)
    tokenizer_from = MarianTokenizer.from_pretrained(model_from)
    model_from = MarianMTModel.from_pretrained(model_from)

    return tokenizer_to, tokenizer_from, model_to, model_from


SRC_TO = "en"
TGT_TO = "roa"

tokenizer_to, tokenizer_from, model_to, model_from = get_models(src=SRC_TO, tgt=TGT_TO)


def translate(
    texts: List[str],
    tokenizer: transformers.models.marian.tokenization_marian.MarianTokenizer,
    model: transformers.models.marian.modeling_marian.MarianMTModel,
):
    translated = model.generate(
        **tokenizer.prepare_seq2seq_batch(texts, return_tensors="pt")
    )
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_text


tokenizer_from.supported_language_codes


def backtranslate(
    text: Union[List, str],
    n_samples: int = 1,
    languages: Union[List[str], str] = "random",
    src_to: str = "en",
    tgt_to: str = "roa",
    specify_language_augmentation=False,
    random_state=42,
    verbose=0,
    filter_out_identical=True,
) -> Tuple[List[str], List[str]]:
    if verbose > 0 and random_state is None:
        print("WARNING: seed is None. ")
    np.random.seed(random_state)

    if SRC_TO == src_to:
        global tokenizer_to
        global tokenizer_from
        global model_to
        global model_from
    else:
        if verbose > 1:
            print("Changing model. ")
        tokenizer_to, tokenizer_from, model_to, model_from = get_models(
            src=src_to, tgt=tgt_to
        )

    assert n_samples <= len(
        tokenizer_to.supported_language_codes
    ), "n_samples > number of supported languages. n_samples = {}, no languages = {}".format(
        n_samples, len(tokenizer_to.supported_language_codes)
    )
    if isinstance(languages, str) is True:
        if languages == "random":
            langs = np.random.choice(
                tokenizer_to.supported_language_codes, size=n_samples, replace=False
            ).tolist()
        else:
            assert (
                n_samples == 1
            ), "You chose one language {}, but n_samples is {}".format(
                languages, n_samples
            )
            langs = list(languages)
    if verbose > 0:
        print("Augmenting using languages: ")
        for lang in langs:
            print(lang)
    translations = []
    texts = [x.strip() for x in text.split(".")]
    for lang in langs:
        if tgt_to in MULTILINGUAL_TGTS:
            to_translate = [
                "{lang} {sentence}".format(lang=lang, sentence=x) for x in texts
            ]
        else:
            to_translate = texts
        print(to_translate)
        # print(model_to)
        # print(model_from)
        translated = translate(to_translate, model=model_to, tokenizer=tokenizer_to)
        if verbose > 1:
            print("Translated: \n{} \nto \n {}".format(to_translate, translated))
        translated_back = translate(
            translated, model=model_from, tokenizer=tokenizer_from
        )
        if verbose > 1:
            print("Translated: \n{} \nto \n {}".format(translated, translated_back))
        translations.append(".".join(translated_back))
    return translations, langs


if __name__ == "__main__":
    _ = backtranslate("This should go to portuguese", n_samples=17, verbose=1)
