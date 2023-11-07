FROM mrt_pytorch_ros_base:latest

RUN sudo apt-get update && sudo apt-get install -y ros-noetic-pcl-ros
RUN pip install scikit-build==0.17.6
RUN pip install pcl-py easydict==1.9 "typer[all]" wget
	
RUN sudo apt-get install -y ros-noetic-jsk-recognition-msgs

ADD . .

CMD /bin/bash -c "source /opt/ros/noetic/setup.bash && cd sfa && python3 ros_node.py"