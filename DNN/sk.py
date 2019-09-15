from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.datasets import load_iris

data = [[1, 3],
        [1, 4],
        [3, 1],
        [4, 1]]
#
y = [1, 1, -1, -1]
model = LogisticRegression(solver='liblinear')
model.fit(data, y)
p = model.predict(np.mat([[1, 4]]))
#
print(p)

# c = load_iris()
# d = c.data
# y = c.target
# print(y)
