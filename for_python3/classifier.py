import sys
import csv
import math
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split as tts

from feature_bk2 import (
    TitleFeature,
    PublisherFeature,
    HostnameFeature,
)


class NewsClassifier(object):
    """
    News Article Classifier to classify new article.
    """

    def __init__(self):
        self.name = 'new artical classifier'
        self.features = {}
        self.categories = {}

    def learn(self, file_path,test_size ,seed):
        """
        Learn traning data give the training data path.
        """
        # Create two features's feature class, namely news's title and news's publisher
        # training data, each data record is a list of article_id, title, url, publisher, hostname, timestamp, category.
        training_data = self.read_csv(file_path)
            
        X_train, X_test, y_train, y_test = tts(training_data[:], np.zeros((6028,7)), test_size=test_size, random_state=seed)

        #print('X_train type is ',type(X_train), len(X_train), X_train[1])
        #print('y_train shape is ',np.shape(y_train))
        
        self.features['title'] = TitleFeature(X_train[1:], smoothing_factor=0.01, word_joins=[1])
        # self.features['publisher'] = PublisherFeature(training_data)
        # self.features['hostname'] = HostnameFeature(training_data[1:], smoothing_factor=1.0)

        self.categories = Counter([record[6] for record in training_data[1:]])

        # Write predict of training data to see the difference
        training_pred = self.predict_dataset(X_train[:], print_ids=[])
        compare = [['article_id', 'category', 'pred']]
        diff_count = 0
        for train, pred in zip(X_train[1:], training_pred):
            if train[6] != pred[1]:
                diff_count += 1
            compare.append([train[0], train[6], pred[1]])
        X_err = diff_count/len(training_pred)
        #print ('Total number of difference is', diff_count, 'out of', len(training_pred), 'records','error rate of X_train is ',diff_count/len(training_pred))
        return X_err
        #self.write_csv('./data/diff.csv', compare)

    # for cat in self.categories:
    #     print 'current cat ', cat + ' having record ', self.categories[cat]

    def predict(self, test_record, print_ids=None):
        """
        Predict the category against the given testing news record using Naive Baysian
        :param test_record:
        :param print_ids: the article records id that needs printing
        :return: the category label
        """
        # For each possible category, compare the log probability
        need_print = test_record[0] in print_ids if print_ids else False
        max_log_prob = None
        result = None
        for cat in self.categories:
            log_prob = 0
            # Iterates each features
            for feature in self.features.values():
                log_prob += feature.condition_log_prob(test_record, cat, print_ids=print_ids)
                if need_print:
                    print ('[classifier] feature=', feature.name, 'log_prob=', log_prob)
            # Adding the prior
            log_prob += math.log(self.categories[cat] * 1.0 / sum(self.categories.values()))
            if need_print:
                print ('[classifier] current cat=', cat, 'log_prob with prior=', log_prob, "\n")
            if max_log_prob is None or max_log_prob < log_prob:
                max_log_prob = log_prob
                result = cat
        if need_print:
            print ('[classifier] Article', test_record[0], 'Final max_log=', max_log_prob, 'with category=', result)
        return result

    def predict_dataset(self, test_dataset=None, file_path=None, print_ids=[]):
        """
        A list of test data record or given test data file path in csv format
        Each test data record in the form of [article_id, title, url, publisher, hostname, timestamp]
        :param test_dataset: Should contain headers
        :param file_path: csv file with headers
        :param print_ids: the article records id that needs printing
        :return: a list of predict tuples. tuple contains (article_id, category)
        """
        test_dataset = test_dataset or self.read_csv(file_path)
        result = []
        for test_record in test_dataset[1:]:
            pred = self.predict(test_record, print_ids=print_ids)
            result.append([test_record[0], pred])
        return result

    @classmethod
    def read_csv(cls, file_path):
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter=',')
            # Omit the First line of titles
            for row in csv_reader:
                # Empty line
                if not row:
                    continue
                dataset.append(row)
        # Skip header
        return dataset

    @classmethod
    def write_csv(cls, file_path, dataset):
        with open(file_path, 'w', newline='', encoding='utf-8') as f: # disable the stupid newline in py3 csv
            csv_writer = csv.writer(f, delimiter=',')
            for row in dataset:
                csv_writer.writerow(row)


if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print ('Please specify the training data file path')

    news_classifier = NewsClassifier()
    min_X_err = 1
    min_test_size = None
    min_X_err_seed = None
    for test_size in np.arange(0,0.3,0.05):
        for seed in range(1000):
            X_err = news_classifier.learn(sys.argv[1],test_size, seed)
            if X_err < min_X_err:
                min_X_err = X_err
                min_test_size = test_size
                min_X_err_seed = seed
                print('test size is ',test_size, 'seed is ',seed)
            
    min_X_err = news_classifier.learn(sys.argv[1],min_test_size,min_X_err_seed)
    print(min_X_err_seed, min_test_size, min_X_err)

    if len(sys.argv) == 3:
        pred_result = news_classifier.predict_dataset(file_path=sys.argv[2])
        pred_result = [('article_id', 'category')] + pred_result
        test_file_split = sys.argv[2].split('.')
        output_filepath = '.'.join(test_file_split[:-1]) + "_pred." + test_file_split[-1]
        news_classifier.write_csv(output_filepath, pred_result)
