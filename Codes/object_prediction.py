#Important thing to know: the message '/darknet_ros/bounding_boxes' contains the informations about all the bounding
#boxes predicted since the node is started till the actual moment. All the following global variables are used to
#extract fistly the incoming bounding boxes, and then only the desired one (the one we are looking at).
#For technical reason we did not use Pupil Core but the webcam, so we decided to consider the object we are looking at
#the one closest to the center.
import rospy
from ros_myo.msg import EmgArray
from std_msgs.msg import String
from darknet_ros_msgs.msg import BoundingBoxes
import numpy as np

#Number that will be used to determine the index from which we can start to read the data from  "bounding_boxes"
global counter
counter=0

#Array that will contain the incoming characteristics of the bounding boxes (xmin,xmax,ymin,ymax)
global container_num
container_num=np.empty([0,4])

#Array that will contain the incoming classes names
global container_class
container_class=np.empty([0,1])

#Dimensions of the webcam screen
width=640
height=480

#Values used to determine the verteces of the bounding boxes
global xmin
global xmax
global ymin
global ymax

global distance #Vectors containing the distances between the incoming bounding boxes and the center. It will be used to determine the bounding box that is the closest to the center
global j #Variable that will be used as index for the incoming bounding boxes (if the incoming are the numbers 104,105,106, we will have j=1,2,3)

#Boolean used to avoid technical problems about the first reading
global first
first = True

xmin=width/2-width/10
xmax=width/2+width/10
ymin=height/2-height/10
ymax=height/2+height/10

def callback(BoundingBoxes): #data are inside msg.data
    dimensions = np.shape(BoundingBoxes.bounding_boxes)
    global counter
    global container_num
    global container_class
    global xmin
    global xmax
    global ymin
    global ymax
    global distance
    global j
    global first
    if first == True:
        fist = False
        counter = dimensions[0]-1
    j = 0
    distance = np.empty(dimensions[0] - counter)

    for i in range(counter, dimensions[0]):
        container_num = np.r_[container_num, np.array([BoundingBoxes.bounding_boxes[i].x, BoundingBoxes.bounding_boxes[i].y, BoundingBoxes.bounding_boxes[i].w, BoundingBoxes.bounding_boxes[i].h]).reshape(1,4)]
        container_class= np.r_[container_class, np.array([BoundingBoxes.bounding_boxes[i].Class]).reshape(1,1)]
        distance[j]=np.linalg.norm(np.array([(container_num[j,0]+container_num[j,2])/2, (container_num[j,1]+container_num[j,3])/2]) - np.array([(xmin+xmax)/2, (ymin+ymax)/2]))
        j = j+1
    pub.publish(container_class[0, np.argmin(distance)])
    container_num = np.empty([0, 4])
    container_class = np.empty([0, 1])
    counter= dimensions[0]



    
	    
rospy.init_node('object_prediction')
pub = rospy.Publisher("Object", String, queue_size=10)
sub_boxes = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, callback)
rospy.spin()
