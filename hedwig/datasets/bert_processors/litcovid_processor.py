"""
EDIT NOTICE

File edited from original in https://github.com/castorini/hedwig
by Bernal Jimenez Gutierrez (jimenezgutierrez.1@osu.edu)
in May 2020
"""

import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class LitCovidProcessor(BertProcessor):
    NAME = 'LitCovid'
    NUM_CLASSES = 8
    IS_MULTILABEL = True
    
    def get_train_examples(self, data_dir, filename='train.tsv'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'LitCovid', filename)), 'train')

    def get_dev_examples(self, data_dir, filename='dev.tsv'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'LitCovid', filename)), 'dev')

    def get_test_examples(self, data_dir, filename='test.tsv'):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'LitCovid', filename)), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = '%s-%s' % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
