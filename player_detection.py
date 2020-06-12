import argparse
import logging
import cv2
import os
from os.path import isfile, join
from numpy import expand_dims
from matplotlib import image, pyplot as plt
from matplotlib.patches import Rectangle
from mrcnn.model import mold_image, MaskRCNN
from tqdm import tqdm
from model_config import LakersJazzConfig
import warnings


def vid_to_frames(video_location, framerate):
    sec = 0
    fps = 1/framerate
    count = 1

    video_capture = cv2.VideoCapture(video_location)
    success = save_frame(video_capture, sec, count)
    while success:
        count = count + 1
        sec = sec + fps
        sec = round(sec, 2)
        success = save_frame(video_capture, sec, count)


def save_frame(vidcap, sec, img_no):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    frames, image = vidcap.read()
    if frames:
        cv2.imwrite('raw_frames/image{0:05d}.jpg'.format(img_no), image)
    return frames


def create_predicted_images(input_dir, output_dir, model, cfg):
    files = [f for f in os.listdir(input_dir) if isfile(join(input_dir, f))]
    for i in tqdm(range(579)):
        jazz_count = 0
        lakers_count = 0
        filename = input_dir + os.sep + files[i]
        img = image.imread(filename)
        scaled_image = mold_image(img, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)[0]
        plt.figure(figsize=(1.66, 0.936), dpi=100)
        plt.imshow(img)
        plt.axis('off')
        ax = plt.gca()
        for box, class_id, score in zip(yhat['rois'], yhat['class_ids'], yhat['scores']):
            y1, x1, y2, x2 = box
            width, height = x2 - x1, y2 - y1
            if class_id == 1:
                rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=0.2)
                ax.text(x1, y1, "lakers", horizontalalignment="left", verticalalignment="bottom", color="white", fontsize=1)
                ax.text(x1, y1, "{:.3f}".format(score), horizontalalignment="left", verticalalignment="top", color="white", fontsize=1)
                lakers_count = lakers_count + 1
            else:
                rect = Rectangle((x1, y1), width, height, fill=False, color='green', linewidth=0.2)
                ax.text(x2, y1, "jazz", horizontalalignment="right", verticalalignment="bottom", color="white", fontsize=1)
                ax.text(x2, y1, "{:.3f}".format(score), horizontalalignment="right", verticalalignment="top", color="white", fontsize=1)
                jazz_count = jazz_count + 1
            ax.add_patch(rect)
        ax.text(1280, 0, "Lakers: {:d}\nJazz: {:d}".format(lakers_count, jazz_count), horizontalalignment="right", verticalalignment="top", color="white", fontsize=2)
        plt.savefig(output_dir + os.sep + files[i], bbox_inches='tight', pad_inches=0, dpi=1000)
        plt.close()


def frames_to_vid(frames_directory, output_name, framerate):
    fps = framerate
    frame_array = []
    files = [f for f in os.listdir(frames_directory) if isfile(join(frames_directory, f))]
    files.sort(key=lambda x: x[5:-4])
    for i in range(len(files)):
        filename = frames_directory + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)

    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Tracks players in a given basketball video and gives an output video of the result.')
    parser.add_argument(
        'videofile', type=str, help='The videofile to be processed.')
    parser.add_argument(
        'framerate', type=int, help='How many frames per second to capture.')
    args = parser.parse_args()

    warnings.simplefilter(action='ignore', category=FutureWarning)

    vid_to_frames(args.videofile, args.framerate)

    cfg = LakersJazzConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)

    model_path = 'cnn_models/mask_rcnn_lj_config_0005.h5'
    model.load_weights(model_path, by_name=True)

    create_predicted_images('raw_frames', 'cnn_frames', model, cfg)
    frames_to_vid('./cnn_frames/', 'videos/output.avi', args.framerate)
