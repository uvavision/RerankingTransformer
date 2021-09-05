from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import datum_io
from delf import feature_io
from delf import utils
from delf.python.detect_to_retrieve import dataset
from delf import extractor

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'delf_config_path', '/tmp/delf_config_example.pbtxt',
    'Path to DelfConfig proto text file with configuration to be used for DELG '
    'extraction. Local features are extracted if use_local_features is True; '
    'global features are extracted if use_global_features is True.')
flags.DEFINE_string(
    'dataset_file_path', '/tmp/gnd_roxford5k.mat',
    'Dataset file for Revisited Oxford or Paris dataset, in .mat format.')
flags.DEFINE_string(
    'images_dir', '/tmp/images',
    'Directory where dataset images are located, all in .jpg format.')
flags.DEFINE_string(
    'output_features_dir', '/tmp/features',
    "Directory where DELG features will be written to. Each image's features "
    'will be written to files with same name but different extension: the '
    'global feature is written to a file with extension .delg_global and the '
    'local features are written to a file with extension .delg_local.')

# Extensions.
_DELG_GLOBAL_EXTENSION = '.delg_global'
_DELG_LOCAL_EXTENSION = '.delg_local'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Read list of images from dataset file.
  print('Reading list of images from dataset file...')
  image_list = read_file(FLAGS.dataset_file_path)
  num_images = len(image_list)
  print('done! Found %d images' % num_images)

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.io.gfile.GFile(FLAGS.delf_config_path, 'r') as f:
    text_format.Parse(f.read(), config)

  # Create output directory if necessary.
  if not tf.io.gfile.exists(FLAGS.output_features_dir):
    tf.io.gfile.makedirs(FLAGS.output_features_dir)

  extractor_fn = extractor.MakeExtractor(config)

  start = time.time()
  missing_images = []
  for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(',')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image_filename = os.path.join(FLAGS.images_dir, image_path)
    if not os.path.exists(input_image_filename):
        missing_images.append(image_name)
        continue
    
    # Compose output file name and decide if image should be skipped.
    should_skip_global = True
    should_skip_local = True
    if config.use_global_features:
      output_global_feature_filename = os.path.join(FLAGS.output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
      if not tf.io.gfile.exists(output_global_feature_filename):
        should_skip_global = False
    if config.use_local_features:
      output_local_feature_filename = os.path.join(FLAGS.output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
      if not tf.io.gfile.exists(output_local_feature_filename):
        should_skip_local = False
    if should_skip_global and should_skip_local:
      print('Skipping %s' % image_name)
      continue

    pil_im = utils.RgbLoader(input_image_filename)
    resize_factor = 1.0
    # if FLAGS.image_set == 'query':
    #   # Crop query image according to bounding box.
    #   original_image_size = max(pil_im.size)
    #   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
    #   pil_im = pil_im.crop(bbox)
    #   cropped_image_size = max(pil_im.size)
    #   resize_factor = cropped_image_size / original_image_size
    im = np.array(pil_im)

    # Extract and save features.
    extracted_features = extractor_fn(im, resize_factor)
    if config.use_global_features:
      global_descriptor = extracted_features['global_descriptor']
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
    if config.use_local_features:
      locations = extracted_features['local_features']['locations']
      descriptors = extracted_features['local_features']['descriptors']
      feature_scales = extracted_features['local_features']['scales']
      attention = extracted_features['local_features']['attention']
      feature_io.WriteToFile(output_local_feature_filename, locations,
                             feature_scales, descriptors, attention)

  print('count', len(missing_images))
  with open('missing.txt', 'w') as f:
    f.write('\n'.join(missing_images))


if __name__ == '__main__':
  app.run(main)
