import re
import math


class Feature(object):
	"""
	Feature class to represent one feature of the data set given the feature index.
	"""

	def __init__(self, name, feature_idx):
		self.name = name
		self.feature_idx = feature_idx

	def condition_log_prob(self, test_record, category):
		"""
		Returning the log conditional probability of passed test_record of current feature given certain category.
		:param test_record: news article record in the form of a list [article_id, title, url, publisher, hostname, timestamp]
		:param category: the class
		:return:
		"""
		raise NotImplementedError()


class TitleFeature(Feature):
	"""
	Feature class representing the title attribute of the data records
	"""

	def __init__(self, training_data):
		"""
		A list of training data records.
		Each record is a list consisting of article_id, title, url, publisher, hostname, timestamp, category.
		:param data:
		"""
		super(TitleFeature, self).__init__('Title', 1)

		# Hold the word count for word in each category.
		self.category_bag_of_words = {}
		# Hold the total word count for each category.
		self.category_count = {}
		for record in training_data:
			category = record[6]
			if self.category_bag_of_words.get(category) is None:
				self.category_bag_of_words[category] = {}
				self.category_count[category] = 0
			bag_of_words = self.category_bag_of_words[category]
			for word in re.split(r"\W", record[1].strip().lower()):
				word = word.strip()
				if not word:
					continue
				if word not in bag_of_words:
					bag_of_words[word] = 0
				bag_of_words[word] += 1
				self.category_count[category] += 1
		# for k, bw in self.category_bag_of_words.items():
		# 	print 'category ', k, ' with number of different words', len(bw)

	def condition_log_prob(self, test_record, category):
		feature_value = test_record[self.feature_idx]
		if category not in self.category_count.keys():
			raise AttributeError('Target category {} does not exist'.format(category))
		log_prob = 0
		for word in re.split(r"\W", feature_value.lower()):
			word = word.strip()
			if not word:
				continue
			word_count = self.category_bag_of_words[category].get(word, 0) + 1
			total_word_count = (len(self.category_count) + self.category_count[category]) # With Laplace estimation
			log_prob += math.log(word_count * 1.0 / total_word_count)
			# print '[title] word=', word, 'word count=', word_count, 'total word count=', total_word_count, 'log_prob=', log_prob
		# print '[title] value=', feature_value, 'log_prob=', log_prob
		return log_prob


class PublisherFeature(Feature):
	"""
	Feature class representing the news publisher attribute of the data records
	"""

	def __init__(self, training_data):
		"""
		A list of training data records.
		Each record is a list consisting of article_id, title, url, publisher, hostname, timestamp, category.
		:param data:
		"""
		super(PublisherFeature, self).__init__('Publisher', 3)
		self.category_bag_of_publishers = {}
		for record in training_data:
			category = record[6]
			if self.category_bag_of_publishers.get(category) is None:
				self.category_bag_of_publishers[category] = {}
			publishers = self.category_bag_of_publishers[category]
			publisher_name = record[3].strip().lower()
			if publisher_name not in publishers:
				publishers[publisher_name] = 0
			publishers[publisher_name] += 1
		# for k, bw in self.category_bag_of_publishers.items():
		# 	print 'category ', k, ' with number of different publisher', len(bw)

	def condition_log_prob(self, test_record, category):
		feature_value = test_record[self.feature_idx]
		if category not in self.category_bag_of_publishers:
			raise AttributeError('Target category {} does not exist'.format(category))
		category_hostname_count = self.category_bag_of_publishers[category].get(feature_value.strip().lower(), 0) + 1
		category_total_count = sum(self.category_bag_of_publishers[category].values())
		log_prob = math.log(category_hostname_count * 1.0 / (len(self.category_bag_of_publishers) + category_total_count))
		# print '[publisher] value=', feature_value, 'count=', category_hostname_count, 'total=', category_total_count, 'log_prob=', log_prob
		return log_prob


class HostnameFeature(Feature):
	"""
	Feature class representing the news hostname attribute of the data records
	"""

	def __init__(self, training_data):
		"""
		A list of training data records.
		Each record is a list consisting of article_id, title, url, publisher, hostname, timestamp, category.
		:param data:
		"""
		super(HostnameFeature, self).__init__('Hostname', 4)
		self.category_bag_of_hostname = {}
		for record in training_data:
			category = record[6]
			if self.category_bag_of_hostname.get(category) is None:
				self.category_bag_of_hostname[category] = {}
			publishers = self.category_bag_of_hostname[category]
			publisher_name = record[4].strip().lower()
			if publisher_name not in publishers:
				publishers[publisher_name] = 0
			publishers[publisher_name] += 1
		# for k, bw in self.category_bag_of_hostname.items():
		# 	print 'category ', k, ' with number of different hostname', len(bw)

	def condition_log_prob(self, test_record, category):
		feature_value = test_record[self.feature_idx]
		if category not in self.category_bag_of_hostname:
			raise AttributeError('Target category {} does not exist'.format(category))
		category_hostname_count = self.category_bag_of_hostname[category].get(feature_value.strip().lower(), 0) + 1
		category_total_count = sum(self.category_bag_of_hostname[category].values())
		log_prob = math.log(category_hostname_count * 1.0 / (len(self.category_bag_of_hostname) + category_total_count))
		# print '[hostname] value=', feature_value, 'count=', category_hostname_count, 'total=', category_total_count, 'log_prob=', log_prob
		return log_prob
