import argparse
import logging
from model_config import LakersJazzDataset, TrainingConfig
from mrcnn.model import MaskRCNN


def prepare_train_set():
    train_set = LakersJazzDataset()
    train_set.initialize('training_set', train=True)
    train_set.prepare()
    return train_set


def prepare_test_set():
    test_set = LakersJazzDataset()
    test_set.initialize('training_set', train=False)
    test_set.prepare()
    return test_set


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Trains the Mask R-CNN model with the given number of epochs. Models are output in cnn_models.')
    parser.add_argument(
        'epochs', type=int, help='How many epochs to use in training.')
    args = parser.parse_args()

    train_set = prepare_train_set()
    test_set = prepare_test_set()

    cfg = TrainingConfig
    cfg.display()

    model = MaskRCNN(mode='training', model_dir='./', config=cfg)
    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    model.train(train_set, test_set, learning_rate=cfg.LEARNING_RATE, epochs=args.epochs, layers='heads')
