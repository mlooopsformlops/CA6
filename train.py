import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("data_processed.csv")

# Prepare the response variable
y = df.pop("cons_general").to_numpy()
y[y < 4] = 0
y[y >= 4] = 1

# Preprocessing features
X = df.to_numpy()
X = preprocessing.scale(X)  # Standard scaling

# Impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)

# Bayesian Ridge Regression
clf = BayesianRidge()
yhat_continuous = cross_val_predict(clf, X, y, cv=5)  # Get continuous predictions
yhat = np.where(yhat_continuous > 0.5, 1, 0)  # Threshold predictions to get binary output

# Calculate metrics
acc = np.mean(yhat == y)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Output metrics to a JSON file
with open("metrics.json", 'w') as outfile:
    json.dump({"accuracy": acc, "specificity": specificity, "sensitivity": sensitivity}, outfile)

# Analyze prediction accuracy across different regions
score = yhat == y
score_int = [int(s) for s in score]
df['pred_accuracy'] = score_int

# Bar plot by region
sns.set_color_codes("dark")
ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette="Greens_d")
ax.set(xlabel="Region", ylabel="Model accuracy")
plt.savefig("by_region.png", dpi=80)
