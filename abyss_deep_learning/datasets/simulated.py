'''Provides some toy datasets
'''
import collections

import numpy as np

from bedrock.utils import text_image


def alphanum_gen(corpus, length, scale=2, thickness=2, noise=5, bg=False, class_probs=None):
    '''Generator that produces images of <length> letters from <corpus> string.
    
    Args:
        corpus (list): A list of ASCII characters that will be used.
        length (TYPE): The number of characters to display per image.
        scale (int, optional): Larger numbers make larger images.
        thickness (int, optional): The thickness of the text.
        noise (int, optional): The standard deviation of the noise to add to the image.
        bg (bool, optional): Add a nonlinear background (buggy).
        class_probs (array, optional): The class probabilities of each element in the corpus.
    
    Yields:
        tuple: An image-label pair.
    '''
    def is_iter(it):
        """Summary
        
        Args:
            it (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return isinstance(it, collections.Iterable)
    if class_probs is None:
        class_probs = np.ones(len(corpus)) / len(corpus)
    while True:
        text = ''.join([str(i) for i in np.random.choice(corpus, p=class_probs, size=length, replace=False)])
        thick = np.random.choice(thickness) if is_iter(thickness) else thickness
        has_bg = np.random.choice(bg) if is_iter(bg) else bg
        image = text_image(text, scale, thickness=thick, noise=noise, bg=has_bg)
        yield image, text


def shapes_gen(max_shapes=1, scale=2, noise=5, class_probs=None, nms=0.3):
    """Generator that procudes images containing various shapes, and instance occupancy masks.
    
    Args:
        max_shapes (int, optional): The maximum number of shapes to have in one image.
        scale (int, optional): A scale parameter deciding how big the image will be.
        noise (int, optional): The standard deviation of pixel noise to be applied.
        class_probs (None, optional): The class probabilities with which to sample the shapes.
        nms (float, optional): Non-maximal suppresion theshold so shapes don't overlap too much.
    
    Yields:
        tuple: (image, shapes, instance_masks, categories)
    """
    import cv2
    from mrcnn.utils import non_max_suppression
        
    def draw_shape(image, shape, dims, color):
        # Get the center x, y and the size s
        x, y, s = dims
        mask = np.zeros(image.shape[0:2], dtype=np.uint8)
        if shape == 'square':
            points = np.array([[
                (x, y),
                (x, y + s),
                (x + s, y + s),
                (x + s, y),
                ]], dtype=np.int32)
            centre = (x + s / 2, y + s / 2)
            angle = np.random.uniform(-180, 180)
            rot_mat = cv2.getRotationMatrix2D(centre, angle, 1)[:2, :2]
#             points = (points @ rot_mat).astype(np.int32)
            image = cv2.fillPoly(image, points, color)
            mask = cv2.fillPoly(mask, points, 1)
        elif shape == "circle":
            image = cv2.circle(image, (x, y), s, color, -1)
            mask = cv2.circle(mask, (x, y), s, 1, -1)
        elif shape == "triangle":
            points = np.array([[(x, y - s),
                                (x - s / np.sin(np.radians(60)), y + s),
                                (x + s / np.sin(np.radians(60)), y + s),
                                ]], dtype=np.int32)
            
            centre = (x + s / 3, y + s / 3)
            angle = np.random.uniform(-180, 180)
            rot_mat = cv2.getRotationMatrix2D(centre, angle, 1)[:2, :2]
#             points = (points @ rot_mat).astype(np.int32)
            image = cv2.fillPoly(image, points, color)
            mask = cv2.fillPoly(mask, points, 1)
        return image, mask.astype(np.bool)

    def random_shape(corpus, height, width):
        # Shape
        from random import choice
        shape = choice(corpus)
        # Color
        color = tuple([np.random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = np.random.randint(buffer, height - buffer - 1)
        x = np.random.randint(buffer, width - buffer - 1)
        # Size
        s = np.random.randint(buffer, height // 4)
        return shape, color, (x, y, s)
    
    def random_image(corpus, height, width):
        # Pick random background color
        bg_color = np.array([np.random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = np.random.randint(1, max_shapes) if max_shapes > 1 else 1
        for _ in range(N):
            shape, color, dims = random_shape(corpus, height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression with threshold to avoid shapes covering each other
        if nms is not None:
            keep_ixs = non_max_suppression(
                np.array(boxes), np.arange(N), nms)
            shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes
    
    corpus = ("triangle", "circle", "square")
    height, width = 50 * scale, 50 * scale
    while True:
        bg_color, shapes = random_image(corpus, height, width)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        instances = np.zeros((height, width, len(shapes)), dtype=np.bool)
        image[...] = bg_color.reshape((1, 1, 3))
        classes = []
        for i, (shape, color, dims) in enumerate(shapes):
            image, instances[..., i] = draw_shape(image, str(shape), dims, color)
            classes.append(corpus.index(shape) + 1)
        if noise:
            image = np.minimum(255, np.maximum(0, 
                np.random.normal(image, noise, size=image.shape))).astype(np.uint8)
        yield image, shapes, instances, classes

