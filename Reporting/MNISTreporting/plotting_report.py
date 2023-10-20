import matplotlib.pyplot as plt
import numpy as np
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
		
print(ig, random)
# Plot the mean accuracies in a line plot
plt.plot(ig['mean'], label='Integrated Gradients')

plt.plot(random['mean'], label='Random Baseline')

# Calculate upper and lower bounds for 'ig' and 'random'
x = range(len(ig['mean']))
ig_upper = np.array(ig['mean']) + np.array(ig['std'])
ig_lower = np.array(ig['mean']) - np.array(ig['std'])
random_upper = np.array(random['mean']) + np.array(random['std'])
random_lower = np.array(random['mean']) - np.array(random['std'])

# Fill between the upper and lower bounds for 'ig' and 'random'
plt.fill_between(x, ig_lower, ig_upper, color='lightblue', alpha=0.5)
plt.fill_between(x, random_lower, random_upper, color='lightcoral', alpha=0.5)
plt.title('Accuracy vs Pixels replaced by mean')
plt.xlabel('Pixels replaced by mean in  %')
plt.ylabel('Accuracy in %')
plt.legend()

plt.xticks(range(len(thresholds)), [int(x*100) for x in thresholds])

# save the plot
path = f"C:/Users/Vik/Documents/4. Private/01. University/2023_Sem6/Intepretable_AI/Latex/PracticalWork/figs/"
plt.savefig(path + 'mean_accuracy_vs_threshold.png')
plt.show()
# plot mean accuracy as

print(ig['std'], random['std'])
