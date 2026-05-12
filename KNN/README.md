# K-Nearest Neighbors : From Scratch with NumPy

A clean implementation of the K-Nearest Neighbors (KNN) classification algorithm built from scratch using NumPy, tested on the Iris flower dataset.

---

## Files

`knn.py` contains the KNN classifier implementation. It computes Euclidean distances between a query point and all training samples, picks the K nearest neighbors, and returns the majority class label as the prediction.

`knntest.py` loads the Iris dataset, splits it into training and test sets, trains the KNN classifier with K=5, runs predictions on the test set, and prints the accuracy.

---

## How It Works

The `KNN` class exposes a familiar scikit-learn style interface : `fit()` stores the training data and `predict()` classifies new samples. For each test point, distances to every training point are computed using Euclidean distance, the K closest neighbors are selected, and the most frequent label among them is returned via majority vote.

The test script uses the built-in Iris dataset from scikit-learn, which contains 150 samples of iris flowers across three species, each described by four features: sepal length, sepal width, petal length, and petal width. An 80/20 train-test split is used for evaluation.

---

## Requirements

```bash
pip install numpy scikit-learn matplotlib
```

---

## Running

```bash
python knntest.py
```

This will print the classification accuracy on the test set. With K=5 and `random_state=1234`, the classifier achieves ~97% accuracy on the Iris dataset.

---

## Adjusting K

To try a different value of K, change the argument when instantiating the classifier in `knntest.py`:

```python
clf = KNN(k=3)   # default is 3 in knn.py
```

Smaller K values are more sensitive to noise; larger K values produce smoother decision boundaries but may underfit.
