from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import utils
import torch
from data.dataset import Dataset

OMNIGLOT_FOLDER = "data/omniglot/"
OMNIGLOT_SPLITS_TRAINVAL = "data/dataset_splits/omniglot/trainval.txt"
OMNIGLOT_SPLITS_TRAIN = "data/dataset_splits/omniglot/train.txt"
OMNIGLOT_SPLITS_VAL = "data/dataset_splits/omniglot/val.txt"
OMNIGLOT_SPLITS_TEST = "data/dataset_splits/omniglot/test.txt"
OMNIGLOT_IMGS_BACKGROUND_ROT = OMNIGLOT_FOLDER + "images_background_resized_rot"
OMNIGLOT_IMGS_EVAL_ROT = OMNIGLOT_FOLDER + "images_evaluation_resized_rot"


class OmniglotDataset(Dataset):

  def __init__(self, nway, split, batch_size):
    super(OmniglotDataset, self).__init__("omniglot", 28, 28, 1, nway, split, batch_size,
                                          "data/omniglot_cache.pklz")
    assert batch_size % nway == 0, "nway must divide batch size"

  def get_batch_points_(self):
    char_classes = self.get_classes_list(self.split, augmented=False)
    n_classes_no_rot = len(char_classes)
    char_inds = np.random.choice(n_classes_no_rot, self.nway, replace=False)
    selected_chars = np.array(char_classes)[char_inds]
    rot_inds = np.random.choice(4, self.nway)
    angles = ['000', '090', '180', '270']
    selected_angles = [angles[i] for i in rot_inds]
    selected_classes = np.array([
        selected_chars[i] + '_rot_' + selected_angles[i]
        for i in range(self.nway)
    ])
    labels_no_rot = np.array([label[:-8] for label in list(self.labels)])
    rot_angles = np.array([label[-3:] for label in list(self.labels)])
    class_point_inds = [] # [[inds for first class], [inds for class second class], ... , ]
    for selected_char, selected_angle in zip(selected_chars, selected_angles):
      sat_char_inds = list(np.where(labels_no_rot == selected_char)[0])
      sat_inds = [
          ind for ind in sat_char_inds if rot_angles[ind] == selected_angle
      ]
      selected_class_inds = np.random.choice(sat_inds, int(self.batch_size / self.nway), replace=False)
      class_point_inds.append(selected_class_inds)

    class_point_inds = np.array(class_point_inds)

    class_img_inds = np.array([self.img_inds[class_inds] for class_inds in class_point_inds])
    selected_labels = np.array([self.labels[class_inds] for class_inds in class_point_inds])
    batch_imgs = np.array([self.load_batch_imgs(img_inds) for img_inds in class_img_inds])
    return batch_imgs, selected_labels

  def get_batch_points(self):
    batch_imgs, selected_labels = self.get_batch_points_()
    return torch.from_numpy((batch_imgs / 255.).astype(np.float32)), selected_labels


  def get_classes_list(self, split, augmented=True):

    def read_splits(fpath, augmented):
      classes = []
      with open(fpath, "r") as f:
        for line in f:
          if len(line) > 0:
            _class = line.strip()
            if augmented:
              for s in ['_rot_000', '_rot_090', '_rot_180', '_rot_270']:
                _class_rot = _class + s
                classes.append(_class_rot)
            else:
              classes.append(_class)
      return classes

    if split == "test":
      classes = read_splits(OMNIGLOT_SPLITS_TEST, augmented)
    elif split == "train" or split == "val":
      if split == "train":
        fpath = OMNIGLOT_SPLITS_TRAIN
      elif split == "val":
        fpath = OMNIGLOT_SPLITS_VAL

      # Check if trainval has already been split into train and val
      if os.path.exists(fpath):
        classes = read_splits(fpath, augmented)
      else:
        # Split trainval into train and val and write to disk
        trainval_classes_norot = read_splits(
            OMNIGLOT_SPLITS_TRAINVAL, augmented=False)
        num_trainval_classes = len(trainval_classes_norot)
        perm = np.arange(num_trainval_classes)
        np.random.shuffle(perm)
        num_training_classes = int(
            0.6 *
            num_trainval_classes)  # 60/40 split of trainval into train/val
        train_classes_inds = list(perm[:num_training_classes])
        val_classes_inds = list(perm[num_training_classes:])
        train_chars = np.array(trainval_classes_norot)[train_classes_inds]
        val_chars = np.array(trainval_classes_norot)[val_classes_inds]

        assert len(train_chars) + len(val_chars) == len(trainval_classes_norot), \
          "Num train chars {} + num val chars {} should be {}.".format(len(train_chars), len(val_chars),
                                                                       len(trainval_classes_norot))
        # Write to disk
        with open(OMNIGLOT_SPLITS_TRAIN, "a") as f:
          for c in train_chars:
            f.write("{}\n".format(c))
        with open(OMNIGLOT_SPLITS_VAL, "a") as f:
          for c in val_chars:
            f.write("{}\n".format(c))
        classes = read_splits(fpath, augmented)
    else:
      raise ValueError(
          "Unknown split. Please choose one of 'train', 'val' and 'test'.")
    return classes

  def load_batch_imgs(self, img_inds):
    batch_imgs = np.array([])
    for i in img_inds:
      img_path = self.images_dict[i]
      img_array = self.load_img_as_array(img_path)
      img_array = img_array.reshape((1, self._channels, self._height, self._width))
      if batch_imgs.shape[0] == 0:  # first image
        batch_imgs = img_array
      else:
        batch_imgs = np.concatenate((batch_imgs, img_array), axis=0)
    return batch_imgs

  def load_dataset(self):
    # Note: The trainval classes are split into training and validation classes
    # by randomly selecting 60 percent of the overall trainval *characters*
    # to be used for training (along with all 4 rotations of each)
    # and the remaining characters (with all their rotations) to be validation classes.
    root_folder_1 = OMNIGLOT_IMGS_BACKGROUND_ROT
    root_folder_2 = OMNIGLOT_IMGS_EVAL_ROT

    # dictionaries mapping index of image into the dataset
    # to the location of the image on disk
    imgs_train_dict = {}
    imgs_val_dict = {}
    imgs_test_dict = {}
    labels_train = []
    labels_val = []
    labels_test = []

    # For example, a class here is: Grantha/character08_rot180 if augmented is true
    train_classes = self.get_classes_list("train", augmented=True)
    val_classes = self.get_classes_list("val", augmented=True)
    test_classes = self.get_classes_list("test", augmented=True)

    example_ind_train = 0
    example_ind_val = 0
    example_ind_test = 0
    num_classes_loaded = -1
    for c in train_classes + val_classes + test_classes:
      num_classes_loaded += 1
      slash_ind = c.find('/')
      alphabet = c[:slash_ind]
      char = c[slash_ind + 1:]

      # Determine which folder this alphabet belongs to
      path1 = os.path.join(root_folder_1, alphabet)
      path2 = os.path.join(root_folder_2, alphabet)
      if os.path.isdir(path1):
        alphabet_folder = path1
      else:
        alphabet_folder = path2

      # The index of the example into the class (there are 20 of each class)
      class_image_num_train = 0
      class_image_num_val = 0
      class_image_num_test = 0

      # char is something like Grantha/character08_rot180
      underscore_ind = char.find('_')
      img_folder = os.path.join(alphabet_folder, char[:underscore_ind])
      rot_angle = char[-3:]  # one of '000', '090', '180', '270'
      img_files = [
          img_f for img_f in os.listdir(img_folder) if img_f[-7:-4] == rot_angle
      ]
      for img in img_files:
        img_loc = os.path.join(img_folder, img)
        if c in train_classes:
          imgs_train_dict[example_ind_train] = img_loc
          example_ind_train += 1
          label = c
          labels_train.append(label)
          class_image_num_train += 1
        elif c in val_classes:
          imgs_val_dict[example_ind_val] = img_loc
          example_ind_val += 1
          label = c
          labels_val.append(label)
          class_image_num_val += 1
        elif c in test_classes:
          imgs_test_dict[example_ind_test] = img_loc
          example_ind_test += 1
          label = c
          labels_test.append(label)
          class_image_num_test += 1
        else:
          raise ValueError("Found a class that does not belong to any split.")
    labels_train = np.array(labels_train)
    labels_val = np.array(labels_val)
    labels_test = np.array(labels_test)
    return imgs_train_dict, labels_train, imgs_val_dict, labels_val, imgs_test_dict, labels_test

  def create_KshotNway_classification_episode(self, K, N):
    all_classes = self.get_classes_list(self.split, augmented=True)
    num_classes = len(all_classes)
    perm = np.arange(num_classes)
    np.random.shuffle(perm)
    chosen_class_inds = list(perm[:N])

    ref_paths, query_paths, labels = [], [], []
    for n in range(N):
      c = all_classes[chosen_class_inds[n]]
      root_folder_1 = OMNIGLOT_IMGS_BACKGROUND_ROT
      root_folder_2 = OMNIGLOT_IMGS_EVAL_ROT
      slash_ind = c.find('/')
      alphabet = c[:slash_ind]
      char = c[slash_ind + 1:]

      # Determine which folder this alphabet belongs to
      # (since the new splits may have mixed which alphabets are background/evaluation)
      # with respect to these folders corresponding to the old splits.
      path1 = os.path.join(root_folder_1, alphabet)
      path2 = os.path.join(root_folder_2, alphabet)
      if os.path.isdir(path1):
        alphabet_folder = path1
      else:
        alphabet_folder = path2

      # char is something like Grantha/character08_rot180
      underscore_ind = char.find('_')
      img_folder = os.path.join(alphabet_folder, char[:underscore_ind])
      rot_angle = char[-3:]  # one of '000', '090', '180', '270'
      img_files = [
          img_f for img_f in os.listdir(img_folder) if img_f[-7:-4] == rot_angle
      ]

      # get an example of this character class
      img_example = img_files[0]  # for example 1040_06_rot_090.png
      char_baselabel = img_example[:img_example.find('_')]  # for example 1040

      # choose K images (drawers) to act as the representatives for this class
      drawer_inds = np.arange(20)
      np.random.shuffle(drawer_inds)
      ref_draw_inds = drawer_inds[:K]
      query_draw_inds = drawer_inds[K:]

      class_ref_paths = []
      class_query_paths = []
      for i in range(20):
        if len(str(i + 1)) < 2:
          str_ind = '0' + str(i + 1)
        else:
          str_ind = str(i + 1)
        img_name = char_baselabel + '_' + str_ind + '_rot_' + rot_angle + '.png'
        class_path = os.path.join(alphabet_folder, char[:underscore_ind])
        img_path = os.path.join(class_path, img_name)
        if i in ref_draw_inds:  # reference
          class_ref_paths.append(img_path)
        elif i in query_draw_inds:  # query
          class_query_paths.append(img_path)

      ref_paths.append(class_ref_paths)
      query_paths.append(class_query_paths)
      labels.append(n)
    return ref_paths, query_paths, labels

  def create_oneshotNway_retrieval_episode(self, N, n_per_class):
    all_classes = self.get_classes_list(self.split, augmented=True)
    num_classes = len(all_classes)
    perm = np.arange(num_classes)
    np.random.shuffle(perm)
    chosen_class_inds = list(perm[:N])

    paths, labels = [], []  # lists of length n_per_class * N
    for n in range(N):
      c = all_classes[chosen_class_inds[n]]
      root_folder_1 = OMNIGLOT_IMGS_BACKGROUND_ROT
      root_folder_2 = OMNIGLOT_IMGS_EVAL_ROT
      slash_ind = c.find('/')
      alphabet = c[:slash_ind]
      char = c[slash_ind + 1:]

      # Determine which folder this alphabet belongs to
      # (since the new splits may have mixed which alphabets are background/evaluation)
      # with respect to these folders corresponding to the old splits.
      path1 = os.path.join(root_folder_1, alphabet)
      path2 = os.path.join(root_folder_2, alphabet)
      if os.path.isdir(path1):
        alphabet_folder = path1
      else:
        alphabet_folder = path2

      # char is something like Grantha/character08_rot180
      underscore_ind = char.find('_')
      img_folder = os.path.join(alphabet_folder, char[:underscore_ind])
      rot_angle = char[-3:]  # one of '000', '090', '180', '270'
      img_files = [
          img_f for img_f in os.listdir(img_folder) if img_f[-7:-4] == rot_angle
      ]

      img_example = img_files[0]  # for example 1040_06_rot_090.png
      char_baselabel = img_example[:img_example.find('_')]  # for example 1040

      perm = np.arange(20)
      np.random.shuffle(perm)
      chosen_drawer_inds = list(perm[:n_per_class])
      for draw_ind, img in enumerate(img_files):
        if not draw_ind in chosen_drawer_inds:
          continue

        if len(str(draw_ind + 1)) < 2:
          str_ind = '0' + str(draw_ind + 1)
        else:
          str_ind = str(draw_ind + 1)

        img_name = char_baselabel + '_' + str_ind + '_rot_' + rot_angle + '.png'
        class_path = os.path.join(alphabet_folder, char[:underscore_ind])
        img_path = os.path.join(class_path, img_name)
        paths.append(img_path)
        labels.append(n)
    return paths, labels

class OmniglotUnsupervised(OmniglotDataset):
  def __init__(self, split, batch_size):
    super(OmniglotUnsupervised, self).__init__(batch_size, split, batch_size)

  def get_batch_points(self):
    ims, ls = super(OmniglotUnsupervised, self).get_batch_points()
    ims = ims.view(self.batch_size, self.channels, self.height, self.width)
    return ims, ls

class OmniglotLoader:
  def __init__(self, nway, split, batch_size):
    self.dataset = OmniglotDataset(nway, split, batch_size)
    self._len = int(len(self.dataset) / batch_size)
  def __len__(self):
    return self._len
  def __getitem__(self, item):
    if item < len(self):
      return self.dataset.get_batch_points()
    else:
      raise IndexError


class OmniglotUnsupervisedLoader:
  def __init__(self, split, batch_size):
    self.dataset = OmniglotUnsupervised(split, batch_size)
    self._len = int(len(self.dataset) / batch_size)
  def __len__(self):
    return self._len
  def __getitem__(self, item):
    if item < len(self):
      return self.dataset.get_batch_points()
    else:
      raise IndexError

