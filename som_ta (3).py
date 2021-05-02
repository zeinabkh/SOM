

import os
from matplotlib import image,pyplot
import numpy as np
dir_path = "G:\master_matus\99_2\\neural network\yalefaces"
images_path = os.listdir(dir_path)
# read each image.
images_data_pix = []
images_label = []
for img_path in images_path:
        images_label.append( img_path[img_path.index(".")+1:].replace(".gif",""))
        try:
           image_data = pyplot.imread(dir_path+"//"+img_path)
           images_data_pix.append( image_data.reshape(-1,image_data.shape[0]*image_data.shape[1])[0]/255)
        except IsADirectoryError :
          continue

X,y = images_data_pix, images_label

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# helper function to plot a color table
def colortable(colors, title, colors_sort = True, emptycols = 0):

	# cell dimensions
	width = 100
	height =20
	swatch_width = 30
	margin = 20
	topmargin = 20

	# Sorting colors bbased on hue, saturation,
	# value and name.
	if colors_sort is True:
		to_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
						name)
						for name, color in colors.items())
		
		names = [name for hsv, name in to_hsv]
		
	else:
		names = list(colors)

	length_of_names = len(names)
	length_cols = 4 - emptycols
	length_rows = length_of_names // length_cols + int(length_of_names % length_cols > 0)

	width2 = width * 4 + 2 * margin
	height2 = height * length_rows + margin + topmargin


	figure, axes = plt.subplots(1,2,figsize =(15,10))

	
	axes[0].set_xlim(0, width * 4)
	axes[0].set_ylim(height * (length_rows-0.5), -height / 2.)
	axes[0].yaxis.set_visible(False)
	axes[0].xaxis.set_visible(False)
	axes[0].set_axis_off()
	axes[0].set_title(title, fontsize = 12, loc ="left", pad = 10)

	for i, name in enumerate(names):
		
		rows = i % length_rows
		cols = i // length_rows
		y = rows * height

		swatch_start_x = width * cols
		swatch_end_x = width * cols + swatch_width
		text_pos_x = width * cols + swatch_width + 7

		axes[0].text(text_pos_x, y, name, fontsize = 10,
				horizontalalignment ='left',
				verticalalignment ='center')

		axes[0].hlines(y, swatch_start_x, swatch_end_x,
				color = colors[name], linewidth = 18)

	return figure,axes

import numpy as np
import math

class SOM:
    def __init__(self, map_size, lr=0.1, sigma=1):
        self.neuron_map = np.random.random((map_size[0], map_size[1], map_size[2]))
        self.neuron_map_count = np.zeros((self.neuron_map.shape[0], self.neuron_map.shape[1], 11))
        self.lr0 = lr
        self.lr = lr
        self.R0 =sigma
        self.R = self.R0
        self.sigma = sigma
        self.js = 0

    def train(self, in_data,y, T=1000, error_threshold=10 ** -20):
        J = []
        for t in range(T):
            pre_neuron_map = np.copy(self.neuron_map)
            random_index = np.random.randint(low=0, high=len(in_data), size=len(in_data))
            for j in random_index:
                x = in_data[j]
                winner = self.get_winner(x)
                net_mask = self.get_NS(winner)
                self.update_weight(x, net_mask)
            # print(pre_neuron_map)
            # print(self.neuron_map)
            j = np.linalg.norm(pre_neuron_map - self.neuron_map)
            J.append(j)
            if t % 10 ==0 :
                 self.visualize(in_data,y)
                 print("epoch:", t, "lr:", self.lr, "J:", j)
            
            self.R = self.R0 * (1-t / 1000)
            self.lr = self.lr0 *  (1-t / T)
            self.js = j
            if j < error_threshold:
                
                break
        self.js = j   
        return J

    def get_winner(self, x):
        copy_x = np.tile(x, [self.neuron_map.shape[0], self.neuron_map.shape[1], 1])
        distance = np.sum((copy_x - self.neuron_map) ** 2, axis=2)
        # print(distance)
        winner = np.unravel_index(np.argmin(distance), shape=distance.shape)
        return winner

    def get_NS(self, winner):
        net_mask = np.zeros((self.neuron_map.shape[0], self.neuron_map.shape[1]))
        wi0 = winner[0]
        wi1 = winner[1]
        for i in range(self.neuron_map.shape[0]):
            for j in range(self.neuron_map.shape[1]):
                net_mask[i, j] = math.exp((- np.linalg.norm(np.array([i, j]) - winner) ** 2) / (2 * self.R ** 2))
        return net_mask

    def update_weight(self, x, net_mask):
        NS = np.tile(net_mask, [self.neuron_map.shape[2], 1, 1]).transpose()
        copy_x = np.tile(x, [self.neuron_map.shape[0], self.neuron_map.shape[1], 1])
        delta = copy_x - self.neuron_map
        # print(NS.shape)
        self.neuron_map = self.neuron_map + np.multiply(NS, delta) * self.lr
    def feature_extract(self,x):
          copy_x = np.tile(x, [self.neuron_map.shape[0], self.neuron_map.shape[1], 1])
          distance = np.sum((copy_x - self.neuron_map) ** 2, axis=2)
          return 1/(1+distance)
    def visualize(self, X, y,t = 0):
        colors = {
            0: [255, 255, 51],
           1: [255, 24, 204],
           2: [250, 240, 177],
            3: [14,70, 10],
            4: [0, 255, 0],
            5: [153, 0, 153],
            6: [255, 0, 0],
            7: [58, 237, 243],
           8: [255, 102, 0],
           9: [224, 224, 224],
            10: [0, 0, 204]
        }
        labels = {
            "sad": 0,
            "sleepy":1,
            "wink": 2,
            "glasses": 3,
            "noglasses": 4,
            "centerlight":5,
            "rightlight": 6,
            "leftlight": 7,
            "happy": 8,
            "normal": 9,
            "surprised": 10
        }
        neuron_map_color = np.zeros((self.neuron_map.shape[0], self.neuron_map.shape[1], 3))
        # self.neuron_map_count = np.zeros((self.neuron_map.shape[0], self.neuron_map.shape[1], 11))
        if t == 1 :
            self.neuron_map_count = np.zeros((self.neuron_map.shape[0], self.neuron_map.shape[1], 11))
        pure_neurons = []
        for i in range(len(X)):
            winner = self.get_winner(X[i])
            self.neuron_map_count[winner[0], winner[1], labels[y[i]]] += 1
        for i in range(self.neuron_map.shape[0]):
            for j in range(self.neuron_map.shape[1]):
                max_n = np.max(self.neuron_map_count[i, j])
                max_index = np.argmax(self.neuron_map_count[i, j])
                un = np.unique(self.neuron_map_count[i, j])
                sum = np.sum(self.neuron_map_count[i, j])
                purity = (max_n / sum) * 100
                if max_n == 0:
                    neuron_map_color[i, j] = [0, 0, 0]
                elif purity >= 50:
                    neuron_map_color[i, j] = colors[max_index]

                elif purity < 50 and purity >= 25:
                    neuron_map_color[i, j] = np.array(colors[max_index]) - 10
                else:
                    neuron_map_color[i, j] = np.array(colors[max_index]) - 14
        f, ax = colortable(colors = {
           "sad": np.array([255, 255, 51])/255,
          "sleepy":  np.array([255, 24, 204])/255,
           "wink":  np.array([250, 240, 177])/255,
           "glasses":  np.array([14, 70, 10])/255,
            "noglasses":  np.array([0, 255, 0])/255,
            "centerlight":  np.array([153, 0, 153])/255,
            "rightlight":  np.array([255, 0, 0])/255,
            "leftlight":  np.array([58, 237, 243])/255,
           "happy":  np.array([255, 102, 0])/255,
           "normal":  np.array([224, 224, 224])/255,
            "surprised":  np.array([0, 0, 204])/255
        }, title="labels",
				colors_sort = False, emptycols = 1)
            
        f.suptitle("R:"+str(self.R0)+"     Learninig_rate:"+str(self.lr0)+"     J:"+ str(self.js) )
        ax[1].imshow(neuron_map_color.astype(np.int64))
        # for k in pure_neurons:
        #   plt.text(k[0,0],k[0,1],list(labels.key())[k[1]],fontsize= 10)
        plt.show()


som = SOM([5,5,X[0].shape[0]],sigma =2.5,lr = 0.1)
J = som.train(in_data=X, y = y, T=1500,error_threshold = 10**-8)
som.visualize(X,y,0)
som.visualize(X,y,1)
plt.plot(J)
plt.ylabel("difference between to weights set")
plt.xlabel("epochs")
plt.show()



