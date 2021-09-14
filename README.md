# SP-ILC
## Introduction
Concurrent Single-Pixel Imaging, Object Location and Classification By Deep Learning.

Paper: **(link)**

We refered to yolov4-tiny implemented in Pytorch when writing the neural network, see https://github.com/bubbliiiing/yolov4-tiny-pytorch. 

## Results Availability
The Test Results of Trained Patterns (sample rate in 333/4096 ~ 8.1%) on Double-and-Triple-MNIST, Fashion-MNIST and our Testset-80 is available in ``Simulation_Results/``. The Results of Random Patterns (sample rate in 333/4096 ~ 8.1%) on the data of our experiments are available in ``Experiment_Results/Testset-80/``.

## Dataset Generating
``dataset_generator/Formtrainingset.m`` is a MATLAB code which is used to form the training set. To change the saving path of the generated images and annotations, see the end of ``dataset_generator/Constructdataset.m``

This code needs MNIST images of JPG form as ingredients, which has been put in ``dataset_generator/train_images.zip``. Unzip the file to get it.

The smaller dataset (with 10% of the total set) generated by the program is available in ``data18k.zip``. Unzip it and follow the next chapter to begin an easy start.

## Before Training
Move the data generated to ``data180k/``.

``data180k/voc2yolo4.py`` is used to transform labels in format ``.xml`` to ``.txt``, with its relative path listed. It is necessary to note that the dataset generator is devoted to create the label in format  ``.xml``. However, we only need the following information of the label at all: ``relative path``, ``xc``, ``yc``, ``width``, ``height``, ``class``. Actually, you can make your own generator and only need to fix ``utils/dataloader.py`` to let it fit your dataset.

``data180k/multA.py`` is used to multiply Matrix A (saved in ``A_8192.txt``) with the images we generated before. We found out that load the image and multiply Matrix A on it for every epoch is not efficient. We sincerely suggest to do it in the beginning and saved the txt files as the input of the network. The results will be saved in ``data180k/``, in order to speed up loading training data.

``data180k/checkdata.py`` is written to ensure that the labels and the images are strictly corresponding, which is useful in debugging.

Other python programs saved in ``data180k/`` is the generator to transfer the dataset into a single file ``.npy``. During training, we read all of the datasets into the memory at once to save the time of repeatedly reading from the hard disk.

The method to rank Hadamard Matrix is available in ``data180k/hadgen.py``.

``model_data/`` saved the settings of Object Locating and classification.

## Training
The main program of the training process is in ``train_180k_(model-name).py``, and the relevant training parameters can be modified in it. The item ``model-name`` is according to our methods demonstrated in the paper(See Results in our paper). The network structure and loss functions are stored in ``netsn/``, and some tool functions are defined in ``utils/``.

We used a image size of 64×64, so the total length of the S sequence saved in our dataset folders is 4096. By modifying the length of the S sequence in ``train_180k.py``, the sampling rate in the actual process can be changed.

The weights as the training results will be saved in ``logs/DATE-TIME/``, where the names of folders as ``DATE-TIME`` is created based on the begining time of training process.

## Predicting
Predicting and Evaluating tools is available in ``eval/``. Files start with ``evaluate`` is the main program for each model, while others are utils. You can choose only to predict or evaluate in these ``evaluate_***.py``. The Results will be saved in the main content, namely ``test_results/``. Total ``PSNR``, ``SSIM``, ``Precision``, and ``Recall`` of the model to be evaluated will be available in ``test_results/results_***.txt``. The details is saved in other files in ``test_results/``

The tools we fit datasets of Double-and-Triple-MNIST, Fashion-MNIST and our Testset-80 to our programs are partially integrated into ``DouTriMnist/``, ``Fashion_MNIST_train/``+``Fashion_MNIST_test/``, and ``expdata_ft`` respectively. Contact us with appropriate requests if you need more details.