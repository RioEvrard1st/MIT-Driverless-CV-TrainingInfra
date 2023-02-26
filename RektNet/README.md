### Description

This is our custom Key Points detection network

## Requirements:

* CUDA>=10.1
* python==3.6
* opencv_python==4.1.0.25
* numpy==1.16.4
* torch==1.1.0
* torchvision==0.3.0
* pandas==0.24.2
* optuna==0.19.0
* Pillow==6.2.1
* protobuf==3.11.0
* pymysql==0.9.3
* tqdm==4.39.0

## Usage
### 1.Download our dataset

##### Download through GCP Tookit
###### 1.1 Image dataset:
```
gsutil cp -p gs://mit-driverless-open-source/RektNet_Dataset.zip ./dataset/
```
then unzip 
```
unzip dataset/RektNet_Dataset.zip -d ./dataset/
```
###### 1.2 Label csv file:
```
gsutil cp -p gs://mit-driverless-open-source/rektnet-training/rektnet_label.csv ./dataset/
```

##### Download manually (Optional)
You can download image dataset and label csv from the link below and unzip them into `./dataset/RektNet_Dataset/` 

[Image dataset](https://storage.cloud.google.com/mit-driverless-open-source/RektNet_Dataset.zip?authuser=1)

[All label csv](https://storage.cloud.google.com/mit-driverless-open-source/rektnet-training/rektnet_label.csv?authuser=1)

### 2.Training

```
python3 train_eval.py --study_name=<name for this experiment>
```

Once you've finished training, you can access the weights file in `./outputs/`

### 3.Inference

#### To download our pretrained Keypoints weights for *Formula Student Standard*, click ***[here](https://storage.googleapis.com/mit-driverless-open-source/pretrained_kpt.pt)***


```
python3 detect.py --model=<path to .pt weights file> --img=<path to an image>
```

Once you've finished inference, you can access the result in `./outputs/visualization/`

#### Run Bayesian hyperparameter search

Before running the Bayesian hyperparameter search, make sure you know what specific hyperparameter that you wish to tuning on, and a reasonable operating range/options of that hyperparameter.

Go into the `objective()` function of `train_hyper.py` edit your custom search

Then launch your Bayesian hyperparameter search
```
python3 train_eval_hyper.py --study_name=<give it a proper name>
```

#### Convert .weights to .onnx manually

Though our training scrip will do automatical .pt->.onnx conversion, you can always do it manually
```
python3 yolo2onnx.py --onnx_name=<path to output .onnx file> --weights_uri=<path to your .pt file>
```

## Code
### cross_ratio_loss.py
This code defines a custom loss function called CrossRatioLoss, which is a subclass of nn.Module in PyTorch. The loss function takes in four arguments: loss_type, include_geo, geo_loss_gamma_horz, and geo_loss_gamma_vert.

The forward() method of this loss function takes in four tensors as inputs: heatmap, points, target_hm, and target_points. The heatmap tensor represents the output of the model and contains the predicted heatmap for the keypoints. The points tensor contains the predicted x,y locations of the keypoints. The target_hm tensor contains the ground truth heatmap for the keypoints, and the target_points tensor contains the ground truth x,y locations of the keypoints.

The function computes the loss based on the selected loss_type. If the loss_type is 'l2_softargmax' or 'l2_sm', it computes the mean squared error (MSE) loss between the predicted and target x,y locations. If the loss_type is 'l2_heatmap' or 'l2_hm', it computes the MSE loss between the predicted and target heatmaps. If the loss_type is 'l1_softargmax' or 'l1_sm', it computes the mean absolute error (MAE) loss between the predicted and target x,y locations.

If include_geo is True, the function also computes a geometric loss based on the co-linearity of the points along the side of the cone and the horizontal lines on the cone. This geometric loss is added to the location loss to get the total loss.

Finally, the function returns three values: the location loss, the geometric loss, and the total loss (which is the sum of the location and geometric losses).

### cspresnet.py
The purpose of this code is to define a Convolutional Neural Network (CNN) architecture called CSPResNet. It is a type of residual network that uses a "cross-stage partial" (CSP) architecture to improve the performance of the model. The CSPResNet consists of two convolutional blocks, each consisting of a sequence of convolutional layers with batch normalization and ReLU activation functions, followed by a final convolutional layer. The output of the first block is concatenated with the output of the second block, and the result is passed through the final convolutional layer to produce the output of the network.

The forward method of the CSPResNet class defines the computation that is performed when the network is run on input data. The input tensor x is passed through the two convolutional blocks, and the outputs of these blocks are concatenated along the channel dimension using the torch.cat function. The concatenated tensor is then passed through the final convolutional layer to produce the output of the network.

The print_tensor_stats function is a utility function that prints the average, minimum, and maximum values of a given tensor, along with a provided name. This function may be useful for debugging or monitoring the training process of a neural network.

### dataset.py
This code defines a custom dataset class called ConeDataset that inherits from the torch.utils.data.Dataset class. The purpose of this dataset class is to provide data for training a neural network model to detect cones in images.

The dataset consists of a set of images and corresponding labels. The images are read from the file system using OpenCV (cv2.imread) and are preprocessed using a prep_image function defined elsewhere. The labels are heat maps that indicate the location of cones in the image, and are also preprocessed using a prep_label function. The labels are also scaled to match the size of the preprocessed image using a scale_labels function.

The ConeDataset class has methods to get the length of the dataset (__len__) and to get a specific data sample (__getitem__). The __getitem__ method returns a tensor representing the preprocessed image, a tensor representing the label heat map, a tensor representing the scaled labels, the name of the original image file, and the size of the original image.

### detect.py
The purpose of this script is to provide a simple interface for running keypoint detection on a single input image using a pre-trained model, and saving the output heatmap image for visualization purposes. The script takes in command-line arguments using argparse and has the following functionality:
1. Loads a pre-trained keypoint detection model from a given filepath.
2. Loads a single input image for keypoint detection from a given filepath.
3. Resizes the input image to a specified size.
4. Performs keypoint detection on the input image using the loaded model.
5. Saves the heat map output of the keypoint detection as an image in the specified output directory.
6. Optionally flips or rotates the input image before performing keypoint detection.

### keypoint_net.py
This code defines a neural network model called KeypointNet that is used for keypoint detection in images. The model takes an image as input and outputs a heatmap that represents the likelihood of each of the seven keypoints being present in the image. The architecture of the network consists of a series of convolutional layers followed by residual blocks and a final convolutional layer that produces the heatmap output.

The print_tensor_stats function is a utility function that prints some statistics about a given tensor, such as its average, minimum, and maximum values. This function is not used in the main functionality of the code.

The KeypointNet class initializes the neural network model with some hyperparameters, such as the number of keypoints to detect, the image size, and whether to run in ONNX mode. The _initialize_weights function sets the initial weights for the neural network layers.

The flat_softmax function applies softmax activation to the flattened heatmap output and reshapes it to the original shape of the heatmap. The soft_argmax function calculates the keypoint locations by computing the weighted average of the x and y coordinates of the heatmap.

The forward function defines the forward pass of the neural network. It takes an input image and passes it through the convolutional layers and residual blocks to produce the heatmap output. If onnx_mode is set to True, it returns the heatmap output as is. Otherwise, it applies the flat_softmax and soft_argmax functions to produce the final keypoint locations.

Finally, in the __main__ block, an instance of KeypointNet is created and a random input image is passed through it to produce an output heatmap. A CrossRatioLoss function is defined and used to calculate the loss between the predicted heatmap and a randomly generated target heatmap.

### keypoint_tutorial_util.py
This code defines several functions to train and evaluate a machine learning model for detecting keypoints on an image. The purpose of this code is to train and evaluate a keypoint detection model, specifically for detecting cones in an image.

The code defines several functions such as print_tensor_stats, eval_model, and print_kpt_L2_distance. print_tensor_stats prints the average, minimum, and maximum values of a tensor. eval_model evaluates the performance of the model on a validation dataset and returns the mean squared error (MSE), geometric loss, and total loss. print_kpt_L2_distance calculates the L2 distance between predicted keypoints and ground truth keypoints, and prints the mean and standard deviation of each keypoint, as well as the total distance error.

The code also initializes the device on which the model will be trained and defines a cuda variable to determine if CUDA is available. Finally, the code writes the best result for the optuna study to a file in the logs directory.

### pt_to_onnx.py
This code defines a Python script that converts a PyTorch trained model with .pt file extension to an ONNX format file with .onnx file extension.

The script uses the KeypointNet class from the keypoint_net module to initialize the model. The KeypointNet model takes 7 as the number of key points and a (80, 80) tuple as the size of the input images. The onnx_mode parameter is set to True to ensure that the model is in ONNX mode.

The main function takes two arguments, the path to the .pt weights file and the name of the output .onnx file, respectively. It loads the trained model using the load_state_dict method of PyTorch and exports it to an ONNX format file using the export method of PyTorch.

The argparse module is used to parse command line arguments for the script. The default value of onnx_name is set to 'new_keypoints.onnx', and weights_uri is required, which specifies the path to the .pt file.

When the script is executed, it checks if it is being executed as the main program using the name == "main" statement, and then calls the main function with the parsed command line arguments. Finally, it prints a message indicating that the conversion has succeeded and saved the .onnx file at the specified location.

### resnet.py
ResNet is a convolutional neural network (CNN) module that implements the residual network (ResNet) architecture, which is commonly used for image classification and other computer vision tasks. The module consists of several convolutional layers with batch normalization and ReLU activation functions, as well as a shortcut connection that allows the gradient to flow directly through the module.

The print_tensor_stats function takes two arguments: a tensor x and a string name. It first flattens the tensor using NumPy, then computes and prints the average, minimum, and maximum values of the flattened tensor. This function can be useful for debugging or monitoring the statistics of tensors during training.

### train_eval.py
The train_model function is responsible for training the model. It takes as input the model architecture, the training data loader, the loss function, the optimizer, the learning rate scheduler, the number of epochs to train for, the validation data loader, the interval for saving checkpoints, the input image size, the number of keypoints, and various other parameters. The function trains the model for the specified number of epochs using the SGD optimizer and updates the learning rate using the scheduler. It also saves the best model checkpoint based on the validation loss.

The eval_model function is responsible for evaluating the model on the validation dataset. It takes as input the model, the validation data loader, the loss function, and the input image size. The function evaluates the model on the validation dataset and returns the validation loss.

The print_tensor_stats function is a utility function for printing the statistics of a PyTorch tensor, such as its mean, minimum, and maximum values.

### train_eval_hyper.py
This code is an implementation of an optimization process using Optuna library to search for the best hyperparameters for a machine learning model.

The code takes in command line arguments using argparse library. It includes the number of trials to run and the name of the study, which is used to track the optimization process.

The objective() function is defined to take in a trial object and uses it to suggest hyperparameters to test. The function then builds an argument list using the suggested hyperparameters, and calls the train_eval.py script with these arguments using the subprocess library. The script trains and evaluates the machine learning model using the suggested hyperparameters and writes the result to a text file. The score is then extracted from the text file and returned as the objective value to minimize by Optuna.

The code creates an Optuna study with the specified name, and then uses the optimize() function to optimize the objective function using the suggested hyperparameters for the specified number of trials.

There are also boolean arguments to initialize studies for vertical and horizontal geometric loss and for three different loss types. These arguments are used to suggest different hyperparameters to test during the optimization process.

Finally, there is an argument to enable automatic instance shutdown after training, which is set to False by default.

### utils.py
This code imports necessary libraries and defines functions and variables for visualization and logging purposes. It also includes functions for image and label preparation.
The variables and functions defined are:
- vis_tmp_path: A string that specifies the temporary directory for visualizations.
- vis_path: A string that specifies the output directory for visualizations.
- Logger: A class that defines a custom logger for writing output to files.
- vis_kpt_and_save: A function that visualizes keypoints and saves the resulting image.
- vis_hm_and_save: A function that visualizes a heat map and saves the resulting image.
- vis_tensor_and_save: A function that visualizes a tensor and saves the resulting image.
- prep_image: A function that prepares an image for processing by resizing it.
- prep_label: A function that prepares a label for processing by creating a heatmap from it.
- get_scale: A function that calculates the scale of an image based on the desired target image size.

The code also deletes any existing vis_tmp_path directory and creates a new one for storing visualizations. The Logger class defines an object that writes output to either sys.stdout or sys.stderr depending on the file extension specified when creating the object. The visualizing functions, vis_kpt_and_save, vis_hm_and_save, and vis_tensor_and_save, create visualizations of keypoints, heatmaps, and tensors respectively and save them to the vis_path directory. The prep_image and prep_label functions prepare images and labels for processing by resizing them and creating a heatmap respectively. The get_scale function calculates the scale of an image based on the desired target image size.
