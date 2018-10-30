import sys
import csv
import math
from collections import Counter

from feature import (
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

	def learn(self, file_path):
		"""
		Learn traning data give the training data path.
		"""
		# Create two features's feature class, namely news's title and news's publisher
		# training data, each data record is a list of article_id, title, url, publisher, hostname, timestamp, category.
		training_data = self.read_csv(file_path)
		self.features['title'] = TitleFeature(training_data[1:], smoothing_factor=0.1)
		# self.features['publisher'] = PublisherFeature(training_data)
		# self.features['hostname'] = HostnameFeature(training_data[1:], smoothing_factor=1.0)

		self.categories = Counter([record[6] for record in training_data[1:]])

		# Write predict of training data to see the difference
		training_pred = self.predict_dataset(training_data[:2])
		print 'test training pred', training_pred
		compare = [['article_id', 'category', 'pred']]
		diff_count = 0
		for train, pred in zip(training_data[1:], training_pred):
			if train[6] != pred[1]:
				diff_count += 1
			compare.append([train[0], train[6], pred[1]])
		print 'Total number of difference is ', diff_count, 'out of ', len(training_pred), 'records'
		self.write_csv('./data/diff.csv', compare)

	# for cat in self.categories:
	# 	print 'current cat ', cat + ' having record ', self.categories[cat]

	def predict(self, test_record):
		"""
		Predict the category against the given testing news record using Naive Baysian
		:param test_record:
		:return: the category label
		"""
		# For each possible category, compare the log probability
		max_log_prob = None
		result = None
		for cat in self.categories:
			log_prob = 0
			# Iterates each features
			for feature in self.features.values():
				log_prob += feature.condition_log_prob(test_record, cat)
				print '[classifier] feature=', feature.name, 'log_prob=', log_prob
			# Adding the prior
			log_prob += math.log(self.categories[cat] * 1.0 / sum(self.categories.values()))
			print '[classifier] current cat=', cat, 'log_prob with prior=', log_prob, "\n"
			if max_log_prob is None or max_log_prob < log_prob:
				max_log_prob = log_prob
				result = cat
		print '[classifier] final max_log=', max_log_prob, 'with category=', result
		return result

	def predict_dataset(self, test_dataset=None, file_path=None):
		"""
		A list of test data record or given test data file path in csv format
		Each test data record in the form of [article_id, title, url, publisher, hostname, timestamp]
		:param test_dataset: Should contain headers
		:param file_path: csv file with headers
		:return: a list of predict tuples. tuple contains (article_id, category)
		"""
		test_dataset = test_dataset or self.read_csv(file_path)
		result = []
		for test_record in test_dataset[1:]:
			pred = self.predict(test_record)
			result.append([test_record[0], pred])
		return result

	@classmethod
	def read_csv(cls, file_path):
		dataset = []
		with open(file_path, 'rb') as f:
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
		with open(file_path, 'w') as f:
			csv_writer = csv.writer(f, delimiter=',')
			for row in dataset:
				csv_writer.writerow(row)


if __name__ == '__main__':
	if len(sys.argv) != 2 and len(sys.argv) != 3:
		print 'Please specify the training data file path'

	news_classifier = NewsClassifier()
	news_classifier.learn(sys.argv[1])

	if len(sys.argv) == 3:
		pred_result = news_classifier.predict_dataset(file_path=sys.argv[2])
		test_file_split = sys.argv[2].split('.')
		output_filepath = test_file_split[0] + "_pred." + test_file_split[1]
		news_classifier.write_csv(output_filepath, pred_result)
