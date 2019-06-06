from abyss_deep_learning.datasets.translators import  AbyssCaptionTranslator, CaptionMapTranslator, AnnotationTranslator
from utils import to_multihot




class HotTranslator(AnnotationTranslator):
    """
    A translator to convert annotations to multihot encoding
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def filter(self, annotation):
        """
        Filters the annotations
        """
        return True
    def translate(self, annotation):
        """
        Translates the annotation into a multihot vector
        Args:
            annotation:

        Returns:

        """
        return to_multihot(annotation, self.num_classes)

class MultipleTranslators(AnnotationTranslator):
    """
    Used when multiple sequential translations are needed to transform the annotations
    """
    def __init__(self, translators):
        for tr in translators:
            assert isinstance(tr, (AnnotationTranslator, type(None)))
        self.translators = translators
    def filter(self, annotation):
        """
        Filters the annotations
        """
        for tr in self.translators:
            if not tr.filter(annotation):
                return False
        return True
    def translate(self, annotation):
        """
        Translates the annotation
        """
        for tr in self.translators:
            annotation = tr.translate(annotation)
        return annotation