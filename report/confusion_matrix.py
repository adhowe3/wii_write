import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Our data collected from live model inference
# expected model inference label mapped with the
# predictions the model gave
data = {
    "A": ["a","a","a","c","i","c","c","c","c","a"],
    "B": ["b","b","b","b","b","b","b","b","b","b"],
    "C": ["c","c","u","c","c","c","c","c","c","c"],
    "D": ["d","d","d","d","d","d","d","d","d","d"],
    "E": ["e","e","e","r","e","e","r","e","e","e"],
    "F": ["f","r","f","f","f","f","f","f","f","f"],
    "G": ["g","g","g","g","g","g","g","d","g","g"],
    "H": ["a","h","h","h","h","h","h","h","h","h"],
    "I": ["i","i","i","i","i","i","i","i","e","i"],
    "J": ["j","j","j","j","j","j","j","j","j","j"],
    "K": ["k","k","k","k","k","k","k","k","k","k"],
    "L": ["l","l","l","l","l","l","l","l","l","l"],
    "M": ["m","m","m","m","m","m","m","m","m","m"],
    "N": ["n","n","n","n","n","n","n","n","n","n"],
    "O": ["o","o","o","o","o","o","o","o","o","o"],
    "P": ["p","p","p","p","p","p","p","p","p","p"],
    "Q": ["q","q","q","c","q","q","q","q","q","q"],
    "R": ["r","r","r","r","b","r","r","r","r","r"],
    "S": ["s","s","s","s","s","s","s","s","s","s"],
    "T": ["t","t","t","t","t","t","t","t","t","t"],
    "U": ["o","u","o","u","u","u","u","u","u","u"],
    "V": ["v","v","v","v","v","v","v","v","v","v"],
    "W": ["w","w","w","w","w","w","w","w","w","w"],
    "X": ["x","x","x","x","x","x","x","x","x","y"],
    "Y": ["y","y","y","y","y","y","y","y","y","y"],
    "Z": ["z","z","z","z","z","z","z","z","z","z"],
}

# Build true and predicted label lists
y_true = []
y_pred = []
total_correct = 0

for true_label, preds in data.items():
    for p in preds:
        y_true.append(true_label.upper())
        y_pred.append(p.upper())  # make them all capital now
        if (p.upper() == true_label):
            total_correct += 1

tot_accuracy = (total_correct) / (26*10)
print(f"total correct: {total_correct}, overall accuracy: {tot_accuracy}")

# Define label order
labels = sorted(list(set(y_true + y_pred)))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues')

# Ticks
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Annotate each cell with count
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j],
                ha="center", va="center")

ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png")