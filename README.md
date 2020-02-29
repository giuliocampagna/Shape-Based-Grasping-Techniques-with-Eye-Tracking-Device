# Shape-Based-Grasping-Techniques-with-Eye-Tracking-Device
The eye tracking algorithm allows to detect our pupil position, and then what our eyes are looking at. Training a YOLO neural network for object recognition, we can recognize the object that we want to grasp. Then a Thalmic’s EMG armband is used to exploit the EMG electric signals coming from 8 electrodes around our forearm in order to understand if the user is closing or opening the hand. If the user wants to close the hand and an object is recognized, the robotic hand will close according to the technique related to the shape of the object (pinch grasp, power grasp or three-jaw chuck grasp).
All the project is performed in Ubuntu 16.04 LTS (Xenial). Basically it has been used Anaconda that is a free and open source distribution of the programming language Python used in Machine Learning, which aims to simplify the management of the packages. Through it we have created a virtual environment within all the necessary libraries (e.g OpenCV). After we have installed Pupil, the myo library (in order
to work with the EMG armband) and darknet that is used as the framework for training YOLO, meaning it sets the architecture of the network. In the end, to connect everything we have installed ROS, an open-source, meta-operating system for robotic systems.



![Overall Setup](master/3DTest.png)
https://github.com/giuliocampagna/Shape-Based-Grasping-Techniques-with-Eye-Tracking-Device/blob/master/Overall_Setup.png
