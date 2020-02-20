# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generic TFX ImportExampleGen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text, Union

import absl
import apache_beam as beam
import tensorflow as tf

from tfx import types
from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.components.example_gen.base_example_gen_executor import INPUT_KEY
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils


CUSTOM_CONFIG_KEY = 'custom_config'
PAYLOAD_FORMAT_KEY = 'payload_format'
DEFAULT_PAYLOAD_FORMAT = example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE


def _PayloadFormat(
    exec_properties: Dict[Text, Any]) -> example_gen_pb2.PayloadFormat:
  """Returns user-specified payload format in exec properties."""
  payload_format = exec_properties.get(PAYLOAD_FORMAT_KEY)
  if not payload_format:
    return DEFAULT_PAYLOAD_FORMAT

  return payload_format


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(Union[bytes, tf.train.Example])
def _ImportRecord(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    input_dict: Dict[Text, List[types.Artifact]],
    exec_properties: Dict[Text, Any],  # pylint: disable=unused-argument
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read TFRecord files to PCollection of recprds (most likely, TF examples).

  Note that each input split will be transformed by this function separately.

  Args:
    pipeline: beam pipeline.
    input_dict: Input dict from input key to a list of Artifacts.
      - input_base: input dir that contains tf example data.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.

  Returns:
    PCollection of TF examples.
  """
  input_base_uri = artifact_utils.get_single_uri(input_dict[INPUT_KEY])
  input_split_pattern = os.path.join(input_base_uri, split_pattern)
  absl.logging.info(
      'Reading input TFExample data {}.'.format(input_split_pattern))

  # TODO(jyzhao): profile input examples.
  records = (
      pipeline
      # TODO(jyzhao): support multiple input format.
      | 'ReadFromTFRecord' >>
      beam.io.ReadFromTFRecord(file_pattern=input_split_pattern))

  if _PayloadFormat(
      exec_properties) == example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE:
    records = (records | 'ToTFExample' >> beam.Map(tf.train.Example.FromString))

  return records


class Executor(BaseExampleGenExecutor):
  """Generic TFX import example gen executor."""

  def PayloadFormat(
      self, exec_properties: Dict[Text, Any]) -> example_gen_pb2.PayloadFormat:
    """Returns user-specified payload format in exec properties."""
    return _PayloadFormat(exec_properties)

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for importing records."""
    print(type(DEFAULT_PAYLOAD_FORMAT))
    return _ImportRecord
