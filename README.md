# Open Space 

## Proof of Concept

### Object Recognition Algorithm
If you want to test this feature, you will need the file object_recognition.py, [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) and [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights). Make sure to give your terminal an access to your webcam. Then, open a terminal from your project repository and use the command:
```
python object_recognition.py
```

*This script has been taken from Murtaza's tutorial. [Here](https://www.youtube.com/watch?v=GGeF_3QOHGE&ab_channel=Murtaza%27sWorkshop-RoboticsandAI) is a link to his Github account.* 

### Streamlit
Here are some demos where you can see features we want to implement to implement in our project. To run them, use commands below inside a Terminal.
This one shows what the final view should looks like:
```
streamlit run demo_streamlit/demo_person_recognition.py
```

There, we tested how to display and parameter a graph using streamlit : 
```
streamlit run demo_streamlit/demo_graph_display.py
```
On this final one (not ready yet), you can display your webcam video on a streamlit page : 
```
streamlit run demo_streamlit/demo_cam_display.py
```
