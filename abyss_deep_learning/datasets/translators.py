'''Provides some translators for datasets'''

from abyss_deep_learning.keras.classification import ClassificationDataset

# Cloudfactory translator
class CloudFactoryCaptionTranslator(ClassificationDataset.AnnotationTranslator):
    '''Translates the CloudFactory labels into a form that works with this script'''
    def filter(self, annotation):
        return 'caption' in annotation and 'type' in annotation and annotation['type'] == 'class_labels'
    def translate(self, annotation):
        return annotation['caption'].split(",")

# Abyss annotation tool translator
class AbyssCaptionTranslator(ClassificationDataset.AnnotationTranslator):
    '''Translates the internal dataset labels into a form that works with this script'''
    def filter(self, annotation):
        return 'caption' in annotation
    def translate(self, annotation):
        return annotation['caption'].split(",")
