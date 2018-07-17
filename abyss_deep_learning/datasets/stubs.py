'''Provides some toy datasets'''
import collections

import numpy as np

from abyss.utils import text_image


def alphanum_gen(corpus, length, scale=2, thickness=2, noise=5, bg=False, class_probs=None):
    '''Generator that produces images of <length> letters from <corpus> string.'''
    def is_iter(it):
        return isinstance(it, collections.Iterable)
    if class_probs is None:
        class_probs = np.ones(len(corpus)) / len(corpus)
    while True:
        text = ''.join([str(i) for i in np.random.choice(corpus, p=class_probs, size=length, replace=False)])
        thick = np.random.choice(thickness) if is_iter(thickness) else thickness
        has_bg = np.random.choice(bg) if is_iter(bg) else bg
        image = text_image(text, scale, thickness=thick, noise=noise, bg=has_bg)
        yield image, text


