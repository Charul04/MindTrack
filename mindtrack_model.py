import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

df = pd.read_csv("mindtrack_dataset_final.csv")

X = df.drop(columns=["Risk_Level"])
y = df["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(classification_report(y_test, y_pred, target_names=["Low Risk", "At-Risk", "High Risk"]))

joblib.dump(model, "mindtrack_model.pkl")

LABELS = ["Low Risk", "At-Risk", "High Risk"]
COLORS = ["#57cc99", "#f4a261", "#e63946"]

# Graph 1: Feature Importance
plt.figure(figsize=(9, 6))
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", color="#7c83fd")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# Graph 2: Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Graph 3: ROC Curves
plt.figure(figsize=(8, 6))
y_bin = label_binarize(y_test, classes=[0, 1, 2])
for i in range(3):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, color=COLORS[i], lw=2, label=f"{LABELS[i]} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()

# Graph 4: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, title in zip(axes, [y_test.values, y_pred], ["Actual", "Predicted"]):
    counts = pd.Series(data).value_counts().sort_index()
    ax.bar(LABELS, counts.values, color=COLORS)
    ax.set_title(title)
    ax.set_ylabel("Count")
plt.suptitle("Actual vs Predicted Distribution")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

