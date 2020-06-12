from xml.etree import ElementTree
from os import listdir
from mrcnn.config import Config
from mrcnn.utils import Dataset
from numpy import zeros, asarray
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class LakersJazzDataset(Dataset):
    def initialize(self, data_directory, train=True):
        self.add_class("dataset", 1, "lakers")
        self.add_class("dataset", 2, "jazz")
        images = data_directory + '/images/'
        annot = data_directory + '/annot/'
        for filename in listdir(images):
            image_id = filename[:-4]
            # skip all images after 150 if we are building the train set
            if train and int(image_id[5::]) >= 80:
                continue
            # skip all images before 150 if we are building the test/val set
            if not train and int(image_id[5::]) < 80:
                continue
            img_path = images + filename
            ann_path = annot + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        count = 0
        info = self.image_info[image_id]
        path = info['annotation']
        l_boxes, j_boxes, w, h = self.extract_player_boxes(path)
        masks = zeros([h, w, len(l_boxes) + len(j_boxes)], dtype='uint8')
        class_ids = list()
        for i in range(len(l_boxes)):
            box = l_boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('lakers'))
            count = count + 1
        for i in range(len(j_boxes)):
            box = j_boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i + count] = 1
            class_ids.append(self.class_names.index('jazz'))
        return masks, asarray(class_ids, dtype='int32')

    def extract_player_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        lakers_boxes = list()
        jazz_boxes = list()
        for box in root.findall('.//object'):
            xmin = int(box.find('bndbox').find('xmin').text)
            ymin = int(box.find('bndbox').find('ymin').text)
            xmax = int(box.find('bndbox').find('xmax').text)
            ymax = int(box.find('bndbox').find('ymax').text)
            coords = [xmin, ymin, xmax, ymax]
            if box.find('name').text == 'lakers':
                lakers_boxes.append(coords)
            else:
                jazz_boxes.append(coords)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return lakers_boxes, jazz_boxes, width, height

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class LakersJazzConfig(Config):
    NAME = "lj_config"
    NUM_CLASSES = 3
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class TrainingConfig(Config):
    NAME = "lj_config"
    NUM_CLASSES = 3
    STEPS_PER_EPOCH = 79



