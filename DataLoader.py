import tensorflow as tf
import time
import numpy as np

class DataLoader(object):
    def __init__(self, data_dir, limits):
        self.train_data_path = [data_dir + '/data/train.summary.id', data_dir + '/data/train.text.id']
        self.test_data_path = [data_dir + '/data/test.summary.id', data_dir + '/data/test.text.id']
        self.dev_data_path = [data_dir + '/data/dev.summary.id', data_dir + '/data/dev.text.id']
        self.limits = limits
        start_time = time.time()

        self.train_set = self.load_data(self.train_data_path)
        self.test_set = self.load_data(self.test_data_path)
        self.dev_set = self.load_data(self.dev_data_path)

        print ('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))

    def load_data(self, path):
        summary_path, text_path = path
        summaries = open(summary_path, 'r').read().strip().split('\n')
        texts = open(text_path, 'r').read().strip().split('\n')
        if self.limits > 0:
            summaries = summaries[:self.limits]
            texts = texts[:self.limits]
        summaries = [list(map(int,summary.split(' '))) for summary in summaries]
        texts = [list(map(int,text.split(' '))) for text in texts]

        return summaries, texts

    def batch_iter(self, data, batch_size, shuffle):
        summaries, texts = data
        data_size = len(summaries)
        num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
            else int(data_size / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            summaries = np.array(summaries)[shuffle_indices]
            texts = np.array(texts)[shuffle_indices]

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            max_summary_len = max([len(sample) for sample in summaries[start_index:end_index]])
            max_text_len = max([len(sample) for sample in texts[start_index:end_index]])
            batch_data = {'enc_in':[], 'enc_len':[], 'dec_in':[], 'dec_len':[], 'dec_out':[]}
            for summary, text in zip(summaries[start_index:end_index], texts[start_index:end_index]):
                summary_len = len(summary)
                text_len = len(text)
                gold = summary + [2] + [0] * (max_summary_len - summary_len)
                summary = summary + [0] * (max_summary_len - summary_len)
                text = text + [0] * (max_text_len - text_len)
                batch_data['enc_in'].append(text)
                batch_data['enc_len'].append(text_len)
                batch_data['dec_in'].append(summary)
                batch_data['dec_len'].append(summary_len)
                batch_data['dec_out'].append(gold)

            yield batch_data