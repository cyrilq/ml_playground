# -*- coding: utf-8 -*-
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from enum import Enum
from csv import reader as csv_reader
from nltk.stem.snowball import RussianStemmer
import pickle as p

stem = RussianStemmer(ignore_stopwords=True)


class FieldOfRequest(Enum):
    spending = 'spending'
    revenue = 'revenue'
    deficit = 'deficit'


def prepare_the_string(string):
    prepared_string = '{}' \
        .format(TextBlob(str(string)
                         .replace('\'', '')
                         .replace('[', '')
                         .replace(']', '')
                         ))
    prepared_string = stem.stem(prepared_string)
    print(prepared_string)
    return prepared_string


train = []

with open('DIR/request_container_spendings.txt', 'r+') as sp_file:
    reader = csv_reader(sp_file)
    for row in reader:
        string_to_append = prepare_the_string(row)
        train.append((string_to_append, FieldOfRequest.spending.name))

with open('DIR/request_container_revenues.txt', 'r+') as re_file:
    reader = csv_reader(re_file)
    for row in reader:
        string_to_append = prepare_the_string(row)
        train.append((string_to_append, FieldOfRequest.revenue.name))

with open('DIR/request_container_deficit.txt', 'r+') as de_file:
    reader = csv_reader(de_file)
    for row in reader:
        string_to_append = prepare_the_string(row)
        train.append((string_to_append, FieldOfRequest.deficit.name))

test = [
    ('сколько денег заработала Чечня в 2011 году', FieldOfRequest.revenue.name),
    ('сколько денег потрачено на экономику в этом году', FieldOfRequest.spending.name),
    ('сколько мы подняли на налогах год назад', FieldOfRequest.revenue.name),
    ('сколько Россия потратит на образование', FieldOfRequest.spending.name),
    ('дефицит Москвы в этом году', FieldOfRequest.deficit.name),
    ('дефицит федерального бюджета 2013', FieldOfRequest.deficit.name),
    ('тверская область расходы на спорт в 2016 году', FieldOfRequest.spending.name)
]

spendings_or_revenues_classifier = NaiveBayesClassifier(train, format='csv')
print(spendings_or_revenues_classifier.accuracy(test))
object = spendings_or_revenues_classifier
file = open('classifier.csv', 'wb')
p.dump(object, file)

new_classifier = NaiveBayesClassifier()