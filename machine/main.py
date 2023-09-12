import pandas as pd
import numpy as np
import sklearn

from sklearn import linear_model
import matplotlib.pyplot as pyplot

import pickle
import matplotlib.style as style

from sklearn.utils import shuffle

data = pd.read_csv("student/student-mat.csv", sep=";")

data = data[["G1", "G2", "studytime", "failures", "absences", "G3"]]

print(sum(data['G3'] > 15))
predict = "G3"

X = np.array(data.drop(columns=[predict]))

y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best_acc = 0  # Initialize the best accuracy

for _ in range(30):
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best_acc:  # Check if the current model is the best so far
        best_acc = acc  # Update the best accuracy
        with open('student_model.pickle', 'wb') as f:
            pickle.dump(linear, f)

print("Best Accuracy:", best_acc)

# Load the best model
pickle_in = open("student_model.pickle", 'rb')
best_linear_model = pickle.load(pickle_in)
print(f"co:{linear.coef_} +intercept : {linear.intercept_}")
prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])
p = ['G1']
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()
