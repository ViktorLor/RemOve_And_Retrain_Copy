import matplotlib.pyplot as plt

path1 = f"C:/Users/Vik/Documents/4. Private/01. University/2023_Sem6/Intepretable_AI/models/mnist/integrated_gradients/reports"
path2 = f"C:/Users/Vik/Documents/4. Private/01. University/2023_Sem6/Intepretable_AI/models/mnist/random_baseline/reports"
# List of file names

ig = {'mean': [], 'std': []}
random ={'mean': [], 'std': []}
for path in [path1, path2]:
	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
	files = [path + f'/{threshold}_report.txt' for threshold in thresholds]
	
	# Empty list to store mean accuracies
	
	# Loop through each file
	for file in files:
		# Open the file and read the accuracy values
		with open(file, 'r') as f:
			lines = f.readlines()
			if path == path1:
				ig['mean'].append(float(lines[6].split(':')[1].strip()[:-1]))
				ig['std'].append(float(lines[5].split(':')[1].strip()[:-1]))
			else:
				random['mean'].append(float(lines[6].split(':')[1].strip()[:-1]))
				random['std'].append(float(lines[5].split(':')[1].strip()[:-1]))
		


# Plot the mean accuracies in a line plot
plt.plot(ig['mean'], label='Integrated Gradients')
# plot the mean accuracies in the same line plot with a different color
plt.plot(random['mean'], label='Random Baseline')
plt.title('Mean Accuracy vs. Threshold')
plt.xlabel('Pixels blocked out in %')
plt.ylabel('Accuracy in %')
plt.legend()
plt.xticks(range(len(thresholds)), thresholds)

# save the plot
path = f"C:/Users/Vik/Documents/4. Private/01. University/2023_Sem6/Intepretable_AI/Latex/figs/"
plt.savefig(path + 'mean_accuracy_vs_threshold.png')
plt.show()
# plot mean accuracy as

# Plot the std accuracies in a table, with the thresholds as the column names and the std accuracies as the values
plt.bar(range(len(thresholds)), ig['std'], label='Integrated Gradients')
plt.bar(range(len(thresholds)), random['std'], label='Random Baseline')
plt.title('Standard Deviation of Accuracy vs. Threshold')
plt.xlabel('Pixels blocked out in %')
plt.ylabel('Standard Deviation of Accuracy in %')
plt.legend()
plt.xticks(range(len(thresholds)), thresholds)
plt.show()