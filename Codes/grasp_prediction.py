#If an object is predicted, an image will be displayed about that object. Then, if it is recognized the grasping of a determined
#object, there will be displayed the correspondent image.
import rospy
import message_filters
from std_msgs.msg import String
import cv2

#Used to display an object only if it is "new" (if the previous one is different)
global prev_Object
prev_Object = ''

#Used to display a grasping only if it is "new" (if the previous one is different)
global prev_EMG
prev_EMG = ''



def callback(String):
    if prev_Object != Object:
        cv2.destroyAllWindows() #Destroy a window if there is it
        if Object == 'Banana':
            cv2.imshow('banana',img) #Show an image with a Banana
            if EMG_prediction == 'Closing' and prev_EMG != 'Closing':
                cv2.destroyAllWindows()
                cv2.imshow('banana_closing',img)
        if Object == 'Orange':
            cv2.imshow('orange',img)
            if EMG_prediction == 'Closing' and prev_EMG != 'Closing':
                cv2.destroyAllWindows()
                cv2.imshow('orange_closing',img)
        if Object == 'Fork':
            cv2.imshow('fork',img)
            if EMG_prediction == 'Closing' and prev_EMG != 'Closing':
                cv2.destroyAllWindows()
                cv2.imshow('fork',img)
            if EMG_prediction == 'Closing' and prev_EMG != 'Closing':
                cv2.destroyAllWindows()
                cv2.imshow('fork_closing',img)
    prev_Object = String.Object
    prev_EMG= EMG_prediction
        
def callback_test(String):
    print(String.Object)


rospy.init_node('grasp_prediction')
sub_test=rospy.Subscriber('/Object',String,callback_test)
pub = rospy.Publisher("Grasp_Prediction", String, queue_size=10)
sub_object = message_filters.Subscriber('/Object', String)
sub_EMG =  message_filters.Subscriber('/EMG_prediction', String)

#We call the callback function only when prediction from EMG and from the object recognition come "almost" together
ts = message_filters.ApproximateTimeSynchronizer([sub_object, sub_EMG], queue_size=10, slop=0.5)
ts.registerCallback(callback)
rospy.spin()
