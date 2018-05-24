FROM nvidia/cuda:8.0-cudnn7-devel

# Expose jupyter notebook and tensorboard ports
EXPOSE 8888
EXPOSE 7001

COPY setup_permissions.sh /tmp/setup_permissions.sh
COPY install-apt-python3 /tmp/install-apt
COPY install-pip /tmp/install-pip
COPY jupyter /home/docker/.jupyter

RUN apt-get update &&  apt-get install -y --fix-missing $(cat /tmp/install-apt)
RUN pip3 install -U pip
# RUN pip install -U $(cat /tmp/install-pip)
RUN pip3 install -U $(cat /tmp/install-pip)
RUN bash /tmp/setup_permissions.sh
RUN apt-get clean && rm -rf /tmp/* /var/tmp/*
RUN echo 'export PYTHONPATH=/home/docker/src/abyss/deep-learning:$PYTHONPATH' >> /home/docker/.profile
RUN jupyter notebook --generate-config

USER docker
ENV HOME /home/docker
CMD jupyter notebook
# pkill -SIGHUP dockerd
