from paraphraser.paraphraser.inference import Paraphraser
import os

STANFORD_PARAPHRASER = Paraphraser(os.path.join("paraphraser", "model", "model-171856"))


def paraphrase(source_sentence, n_examples, sampling_temp):
    pass
