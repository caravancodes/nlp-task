from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import LinearSVC
from collections import Counter
from sklearn.model_selection import train_test_split

import csv, time
import numpy as np

data_set_name = "exasens.csv"
source = "https://archive.ics.uci.edu/ml/datasets/Exasens"
data = []
label = []
with open(data_set_name, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        data.append(row[4])
        label.append(row[3])

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=123)

start = time.time()

clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC())])
clf.fit(x_train, y_train)

new_data = ['76']

pred = clf.predict(new_data)
pred_2 = clf.predict(x_test)
accuracy = np.mean(pred_2 == y_test)

end = time.time()
print("Data Set \t\t: {}".format(data_set_name))
print("Sumber data \t: {}".format(source))
print("Jumlah data \t: {}".format(len(data)))
print("Data training \t: {}".format(len(x_train)))
print("Data testing \t: {}".format(len(x_test)))
print("Hasil prediksi \t: {}".format(pred))
print("Akurasi \t\t: {}".format(accuracy))
print("Waktu \t\t\t: {}".format(end - start))

print()
print("Counter jumlah data :")
print(Counter(label))  # Counter Jumlah Data

print()
print("Counter data training ")
print(Counter(y_train))  # Counter Data Training

print()
print("Counter data testing")
print(Counter(y_test))  # Counter Data Testing
