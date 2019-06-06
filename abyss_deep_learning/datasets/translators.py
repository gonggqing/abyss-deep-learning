'''Provides the base translator class, and some specific translators for datasets.'''

class AnnotationTranslator(object):
        '''Base class for a functor that transforms annotations from a source dataset and task
            to a target task.
        '''
        def filter(self, annotation):
            '''Filter function for annotations.
            Useful to filter out non-task related annotations.

            Args:
                annotation: An annotation object, structure is dataset dependent.

            Returns:
                boolean: Whether or not to use this annotation.
            '''
            return True

        def translate(self, annotation):
            '''Transform the annotation for use with a new task.

            Args:
                annotation: An annotation object, structure is dataset and task dependent.

            Returns:
                The transformed annotation.
            '''
            return annotation


class CocoCaptionTranslator(AnnotationTranslator):
    '''Translates the internal dataset labels into a form that works with this script'''

    def __init__(self, separator=None):
        """Instantiate a translator that reads standard COCO captions as a string.
        Optionally separate fields by a delimiter.

        Args:
            separator (str, optional): A delimiter to separate fields
        """
        self.separator = separator

    def filter(self, annotation):
        '''Filter out non-caption annotations'''
        return 'caption' in annotation

    def translate(self, annotation):
        '''Return a list of strings'''
        caption = annotation['caption']
        return caption.split(self.separator) if self.separator else [caption]


class AbyssCaptionTranslator(AnnotationTranslator):
    '''Translates the Abyss Annotation Tool labels into a list of strings.'''

    def filter(self, annotation):
        '''Filter out non-caption annotations'''
        return 'caption' in annotation and 'type' in annotation and annotation['type'] == 'class_labels'

    def translate(self, annotation):
        '''Return a list of strings'''
        return annotation['caption'].split(",")


### Examples
class CaptionMapTranslator(AnnotationTranslator):
    '''Translates the CloudFactory labels into a form that works with this script'''
    def __init__(self, mapping):
        """Map source -> target annotations.
            e.g.:
                mapping={word: number for number, word in enumerate(list("abcdefg"))}
                >> {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6}

        Args:
            mapping (dict): A dictionary mapping source keys to target values.
        """
        self.map = mapping

    def filter(self, annotation):
        return 'caption' in annotation

    def translate(self, annotation):
        # return [self.map[annotation['caption']]]
        return [self.map[caption] for caption in annotation['caption'].split(',')]


class CategoryTranslator(AnnotationTranslator):
    """
    Gets the category id from the annoations and translates this to logits
    """
    def __init__(self, mapping):
        """

        Args:
            mapping: key=category_id, value=logit_id. e.g. {1: 0, 2: 1, 3: 2}

        """
        self.map = mapping

    def filter(self, annotation):
        return annotation['category_id'] in list(self.map.keys())

    def translate(self, annotation):
        """
        Translates the annotation to logits
        Args:
            annotation: (dict) the annotation dictionary, as retrieved from the coco dataset

        Returns:
            (list) A list of logit ids


        """
        return [self.map[annotation['category_id']]]


