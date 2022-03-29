# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A demo script showing how to use the uisrnn package on toy data."""

import numpy as np

import uisrnn


SAVED_MODEL_NAME = 'saved_model.uisrnn'


def diarization_experiment(model_args, training_args, inference_args):
  """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """

  predicted_cluster_ids = []
  test_record = []

  #train_data = np.load('./data/toy_training_data.npz', allow_pickle=True)
  #test_data = np.load('./data/toy_testing_data.npz', allow_pickle=True)
  #train_sequence = train_data['train_sequence']
  #train_cluster_id = train_data['train_cluster_id']
  #test_sequences = test_data['test_sequences'].tolist()
  #test_cluster_ids = test_data['test_cluster_ids'].tolist()
  train_sequence = np.load('./data/train_sequence.npy', allow_pickle=True)
  train_cluster_id = np.load('./data/train_cluster_id.npy', allow_pickle=True)
  test_sequences = np.load('./data/test_sequence.npy', allow_pickle=True)
  test_cluster_ids = np.load('./data/test_cluster_id.npy', allow_pickle=True)


  # Apply test set splititing from (https://github.com/HarryVolek/PyTorch_Speaker_Verification/issues/17)
  test_seq_2 = []
  test_id_2 = []

  batch_seq = []
  batch_id = []

  count = 0
  for test_sequence, test_cluster_id in zip(test_sequences, test_cluster_ids):
    batch_seq.append(test_sequence)
    batch_id.append(test_cluster_id)

    count = count + 1
    if count != 0 and count % 100 == 0:
        test_seq_2.append(batch_seq)
        batch_seq = []

        test_id_2.append(batch_id)
        batch_id = []

    np.save('./data/new_test.npy', test_sequences)

    # (45,100,256) -> [array(100,256), array(100,256) .... ]
    test_seq_2 = np.empty(len(test_seq_2), object)
    test_seq_2[:] = [np.array(a) for a in test_seq_2]

    test_sequences = test_seq_2
    test_cluster_ids = test_id_2




  model = uisrnn.UISRNN(model_args)

  # Training.
  # If we have saved a mode previously, we can also skip training by
  # calling：
  # model.load(SAVED_MODEL_NAME)
  model.fit(train_sequence, train_cluster_id, training_args)
  model.save(SAVED_MODEL_NAME)

  # Testing.
  # You can also try uisrnn.parallel_predict to speed up with GPU.
  # But that is a beta feature which is not thoroughly tested, so
  # proceed with caution.
  # for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
  print(f'test_sequence.ndim: {test_sequence.ndim}')
  print("%" * 100)
  print(test_sequence.shape, type(test_sequence))
  print(test_cluster_id, type(test_cluster_id))

  predicted_cluster_id = model.predict(test_sequence, inference_args)
  predicted_cluster_ids.append(predicted_cluster_id)
  accuracy = uisrnn.compute_sequence_match_accuracy(
      test_cluster_id, predicted_cluster_id)
  test_record.append((accuracy, len(test_cluster_id)))
  print('Ground truth labels:')
  print(test_cluster_id)
  print('Predicted labels:')
  print(predicted_cluster_id)
  print('-' * 80)

  output_string = uisrnn.output_result(model_args, training_args, test_record)

  print('Finished diarization experiment')
  print(output_string)


def main():
  """The main function."""
  model_args, training_args, inference_args = uisrnn.parse_arguments()
  diarization_experiment(model_args, training_args, inference_args)


if __name__ == '__main__':
  main()
