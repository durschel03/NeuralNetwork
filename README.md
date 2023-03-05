# NeuralNetwork
This program is a customizable multi layer back propagation neural network "from scratch".

  This is a customizable implementation of a classic back propagation (gradient descent) neural network with a learning rate, a gain and a momentum factor. The network also has stopped learning functionality, meaning that when the network continues learning until it reaches a local minimum, at this point the loop is broken and the network is tested. The training function uses an array in the program that contains 280 years of SunSpot data. Part of the array is used for training, another part is used for testing and another part is used for network evaluation. The network has three layers total. The input layer has 30 neurons and takes 30 years of sunspot data. The hidden layer contains ten neurons, however more hidden layers and neurons could easily be added by modifying the NUM_LAYERS & Units[NUM_LAYERS] variables at the top. The program trains by repeatedly selecting random years within the TRAIN_LWB to TRAIN_UPB variable range. The output layer has one neuron which represents the network's prediction for the following year's sunspot activity after the initial 30 years that the input layer accepted. When modifying input data and adding more hidden layers it is also important to add a for loop on line 427 & line 428 to add every output neuron's value to the closed loop data array (in that case the SunSpot_ data array would also be structured differently). 


Here is the mathematical framework that I used to build the network:

![Diagram](https://user-images.githubusercontent.com/126504870/222942059-86d69bcf-20ac-48db-954c-067017c69012.jpg)

For more information visit: https://en.wikipedia.org/wiki/Backpropagation
