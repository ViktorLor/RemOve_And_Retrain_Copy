
import matplotlib
import matplotlib.pyplot as plt


# plot 5 pictures side by side
path = f'C:\\Users\\Vik\\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\Latex\\figs'
fig, ax = plt.subplots(2, 5, figsize=(20, 4))

for i in [1,3,5,7,9]:
	#load image from path
	img = plt.imread(f'{path}\\IG0.{i}.png')
	
	#plot image
	ax[0, int(i/2)].imshow(img)
	ax[0, int(i/2)].set_title(f'0.{i}')

for i in [1, 3, 5, 7, 9]:
	# load image from path
	img = plt.imread(f'{path}\\R0.{i}.png')
	
	# plot image
	ax[1, int(i / 2)].imshow(img)

# plot title for row
ax[0, 0].set_ylabel('Integrated Gradients')
ax[1, 0].set_ylabel('Random')

# make subplots close to each otehr
fig.subplots_adjust(wspace=0.01)
# make subplotsclose to
# plot 5 pictures side by side
plt.show()