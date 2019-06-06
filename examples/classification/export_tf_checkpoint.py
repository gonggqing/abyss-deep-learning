import argparse
import keras.backend as K
import keras
import tensorflow as tf

from abyss_deep_learning.keras.models import ImageClassifier

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
    This script is to show how to visualise the data
    """)
    parser.add_argument("model_path", type=str, help="Path to the coco dataset")
    parser.add_argument("output_ckpt", type=str, help="Path to keras model")
    args = parser.parse_args()
    return args

def main(args):
    classifier = ImageClassifier.load(args.model_path)
    K.set_learning_phase(0)

    print(classifier.model_.output.op.name)

    saver = tf.train.Saver()
    saver.save(K.get_session(), args.output_ckpt)

if __name__ == "__main__":
    main(get_args())