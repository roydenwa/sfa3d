FROM gitlab.mrt.kit.edu:21443/pub/pytorch_ros_deployment/mrt_pytorch_ros_base:latest

RUN sudo apt-get update && sudo apt-get install -y ros-noetic-pcl-ros ros-noetic-jsk-recognition-msgs
RUN pip install scikit-build==0.17.6
RUN pip install pcl-py easydict==1.9 "typer[all]" wget

ADD . .

ENV LOG_LEVEL 2
ENV NODE_NAME "sfa3d_detector"
CMD /bin/bash -c "source /opt/ros/noetic/setup.bash && cd sfa && python3 ros_node.py --log-level ${LOG_LEVEL} --node-name ${NODE_NAME}"