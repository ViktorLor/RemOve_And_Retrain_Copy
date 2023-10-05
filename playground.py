# Import necessary libraries
from sklearn import tree
import matplotlib.pyplot as plt

# Define the example dataset
X = [[2.5, 3.0], [1.0, 2.0], [4.0, 2.5], [3.5, 4.0], [5.0, 3.5], [4.5, 1.5]]
y = [0, 0, 1, 1, 0, 1]



# Create a decision tree classifier
clf = tree.DecisionTreeClassifier()

# Fit the classifier to the dataset
clf = clf.fit(X, y)



# Plot the decision tree
plt.figure(figsize=(9, 6))

# Subplot for the dataset table
plt.subplot(121)
plt.axis('off')  # Turn off axis for the table
plt.title("Dataset")
# plot X and y
# join X and y into a single list of lists
data = [x + [y[i]] for i, x in enumerate(X)]
#plot the table
plt.table(cellText=data, colLabels=['Feature 1', 'Feature 2', 'Class'], loc='center')


# Subplot for the decision tree
plt.subplot(122)
tree.plot_tree(clf, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'])
plt.title("Decision Tree")

plt.tight_layout()

# save plot
plt.savefig('decision_tree.png')