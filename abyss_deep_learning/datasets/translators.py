'''Provides some translators for datasets'''

class AnnotationTranslator(object):
        '''Base class to transform annotations into list of captions.'''
        def filter(self, annotation):
            '''Whether or not to use a annotation'''
            return 'caption' in annotation
        def translate(self, annotation):
            '''Transform the annotation in to a list of captions'''
            return [annotation['caption']]

# Cloudfactory translator
class CloudFactoryCaptionTranslator(AnnotationTranslator):
    '''Translates the CloudFactory labels into a form that works with this script'''
    def filter(self, annotation):
        return 'caption' in annotation and 'type' in annotation and annotation['type'] == 'class_labels'
    def translate(self, annotation):
        return annotation['caption'].split(",")

# Abyss annotation tool translator
class AbyssCaptionTranslator(AnnotationTranslator):
    '''Translates the internal dataset labels into a form that works with this script'''
    def filter(self, annotation):
        return 'caption' in annotation
    def translate(self, annotation):
        return annotation['caption'].split(",")
