# Autonomous-Driving-Car-simulation-using-Pytorch

## Contributors-

1.Shreyas.P.J , github: shreyaspj20

# ABSTRACT:

A self-driving car, also known as an autonomous vehicle (AV or auto), driverless car, or robo-car is a vehicle that is capable of sensing its environment and moving safely with little or no human input. In this project, we will be using camera sensors(center,left and right) implanted on the car to make predictions on the steering angle.This steering angle will be used to control the car. We will start of with building a model with the architecture which is defined in the NVIDIA reasearch paper https://arxiv.org/pdf/1604.07316v1.pdf. This model was trained on 22 epochs. The model could perform better if it is trained on a lot more training images. Augumentation of images could also help reduce overfitting of the model.

 
 # IMPORTANT LIBRARIES USED:
1.OpenCV.

2.Torch.

3.Torchvision.

4.Socketio.

5.Eventlet.

6.Pillow.


# SIMULATOR USED.
To test how our model actually works, we need a simulator to actually run our trained model on. We can run simulations for such models on CARLA or Grand Theft Auto(GTA) but we will be using an open sourced simulator by Udacity. One can find the documentation and download the beta version simulator from https://github.com/udacity/self-driving-car.
This has 2 modes:
1. Training mode: We have to generate train images by use of manual driving. The train images are generated along with the driving csv file which contains the path of each image captured and also the steering angle at each instance.
2. Autonomous mode: After we are done with training our model, we can test it by deploying the model on the simulator.


![](Self_Driving_Car_Simulation.gif)

# HOW TO USE
1. Install all the required dependencies defined in the requirements.txt.
2. This application can be used by executing the run.py file.
3. The model could be run by executing this command on the terminal  "python run.py model.h5".
4. If you are using your own model, then replace model.h5 with your model's path in the command.

Enjoy the ride :)

