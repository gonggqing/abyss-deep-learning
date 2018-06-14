'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, masks) and not in a batch.
Masks may be either categorical or binary.
'''

import numpy as np

