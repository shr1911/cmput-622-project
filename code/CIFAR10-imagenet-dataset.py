import glob
import os

import libarch
import libdata
import libutil

import numpy as np
import sonnet as snt
import tensorflow as tf

from tqdm import tqdm


def load_data(name, provider='tfds.TFDSImagesNumpy', kwargs=None):
  kwargs = kwargs or dict()
  kwargs['name'] = name
  ctor = libutil.rgetattr(libdata, provider)
  return ctor(**kwargs)


def build_model(num_classes, arch, kwargs=None):
  kwargs = kwargs or dict()
  kwargs['num_classes'] = num_classes
  ctor = libutil.rgetattr(libarch, arch)
  return ctor(**kwargs)


def load_checkpoint(model, checkpoint_dir):
  v_epoch = tf.Variable(0, dtype=tf.int64, name='epoch', trainable=False)
  v_gs = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
  checkpoint = tf.train.Checkpoint(model=model, epoch=v_epoch, global_step=v_gs)
  
  ckpt_list = glob.glob(os.path.join(checkpoint_dir, 'ckpt-*.index'))
  assert len(ckpt_list) == 1
  ckpt_path = ckpt_list[0][:-6]
  checkpoint.restore(ckpt_path).expect_partial()
  return dict(epoch=int(v_epoch.numpy()), global_step=int(v_gs.numpy()), path=ckpt_path)


def do_eval(model, dataset, split='test', batch_size=200):
  correctness_all = []
  index_all = []
  for inputs in tqdm(
      dataset.iterate(split, batch_size, shuffle=False, augmentation=False),
      total=int(dataset.get_num_examples(split) / batch_size)):
    predictions = model(inputs['image'], is_training=False)
    correctness_all.append(tf.equal(tf.argmax(predictions, axis=1),
                                    inputs['label']).numpy())
    index_all.append(inputs['index'].numpy())

  correctness_all = np.concatenate(correctness_all, axis=0)
  index_all = np.concatenate(index_all, axis=0)

  return dict(correctness=correctness_all, index=index_all)


def run_demo(model_dir, arch, dataset, split):
  model = build_model(dataset.num_classes, arch)
  load_results = load_checkpoint(model, os.path.join(model_dir, 'checkpoints'))

  aux_arrays = np.load(os.path.join(model_dir, 'aux_arrays.npz'))
  subsample_tr_idx = aux_arrays['subsample_idx']
  print(f'Loaded from checkpoint (epoch={load_results["epoch"]}, ' +
        f'global_step={load_results["global_step"]}) trained from a random ' +
        f'{len(subsample_tr_idx)/dataset.get_num_examples("train")*100:.0f}%' +
        ' subset of training examples.')

  results = do_eval(model, dataset, split=split)
  print(f'Eval accuracy on {split} = {np.mean(results["correctness"]):.4f}')

  def ordered_correctness(correctness, index):
    new_correctness = np.zeros_like(correctness)
    new_correctness[index] = correctness
    return new_correctness

  oc1 = ordered_correctness(results['correctness'], results['index'])

  if split == 'train':
    exported_correctness = np.concatenate(
      [aux_arrays['correctness_train'], aux_arrays['correctness_removed']], axis=0)
    exported_index = np.concatenate(
      [aux_arrays['index_train'], aux_arrays['index_removed']], axis=0)
  else:
    exported_correctness = aux_arrays[f'correctness_{split}']
    exported_index = aux_arrays[f'index_{split}']
  oc2 = ordered_correctness(exported_correctness, exported_index)
  n_match = np.sum(oc1 == oc2)
  print(f'{n_match} out of {len(oc1)} predictions matches')


def cifar10_demo(model_dir, arch='inception.SmallInception', split='test'):
  dataset = load_data('cifar10:3.0.2')
  run_demo(model_dir, arch, dataset, split)


def imagenet_demo(model_dir, arch='resnet_sonnet.ResNet50', split='test'):
  dataset = load_data('imagenet', provider='indexed_tfrecords.IndexedImageDataset')

  run_demo(model_dir, arch, dataset, split)


if __name__ == '__main__':
  cifar10_demo('/path/cifar10-inception/0.5/123')
  # imagenet_demo('/path/imagenet-resnet50/0.7/123')
