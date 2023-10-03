import numpy as np
import matplotlib.pyplot as plt

# original filename
file_names = [
	r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\models\food101\paper_results.txt',
	r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\models\food101\results.txt']

# Initialize data dictionaries
data = {
	"Random_Baseline": {},
	"Integrated_Gradient": {},
	"Guided_Backprop": {}
}

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
width = 0.2

# Read and extract data from each file
for file_name in file_names:
	with open(file_name, "r") as file:
		lines = file.readlines()
		
		current_category = None
		
		for line in lines:
			line = line.strip()
			if line == "Random_Baseline:":
				current_category = "Random_Baseline"
			elif line == "Integrated_Gradient:":
				current_category = "Integrated_Gradient"
			elif line == "Guided_Backprop:":
				current_category = "Guided_Backprop"
			elif ":" in line:
				key, value = line.split(":")
				data[current_category][key.strip()] = float(value.split('%')[0].strip())  # Remove '%' symbol

	
	if file_name == file_names[0]:
		a = ax[0]
		a.set_title('Accuracy of the original model')
		
	else:
		a = ax[1]
		a.set_title('Accuracy of the modified model')
		
	
	x = np.arange(len(data["Random_Baseline"]))
	# make it a line plot
	a.plot(x, data["Random_Baseline"].values(), label="Random Baseline")
	a.plot(x, data["Integrated_Gradient"].values(), label="Integrated Gradient")
	a.plot(x, data["Guided_Backprop"].values(), label="Guided Backprop")
	a.set_xlabel('Pixels removed in %')
	a.set_ylabel('Accuracy in %')
	
	# set the y axis from 50 to 100
	a.set_ylim([50, 100])

	a.set_xticks(x)
	a.set_xticklabels(data["Random_Baseline"].keys())
	a.legend()
plt.show()