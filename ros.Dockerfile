FROM gitlab.mrt.kit.edu:21443/pub/pytorch_ros_deployment/mrt_pytorch_ros_base:latest

RUN sudo apt-get update && sudo apt-get install -y ros-noetic-pcl-ros ros-noetic-jsk-recognition-msgs
RUN pip install scikit-build==0.17.6
RUN pip install pcl-py easydict==1.9 "typer[all]" wget omegaconf==2.3.0

ADD . .

RUN cd /workspace/catkin_ws && catkin build
CMD /bin/bash -c "source /opt/ros/noetic/setup.bash && source /workspace/catkin_ws/devel/setup.bash && roslaunch sfa3d sfa3d.launch"