FROM nvidia/cuda:9.0-cudnn7-devel
# Possibly cuda:9.0-cudnn7-devel-ubuntu16.04

# Expose jupyter notebook and tensorboard ports
EXPOSE 8888
EXPOSE 7001

COPY setup.sh /tmp/setup.sh
COPY install-pip /tmp/install-pip
COPY install-apt-python3 /tmp/install-apt

RUN bash /tmp/setup.sh prerequisites

COPY ssh /home/docker/.ssh
RUN chown -R docker /home/docker/.ssh
RUN chgrp -R docker /home/docker/.ssh
COPY git-clone-with-key /home/docker/bin
RUN bash /tmp/setup.sh local-packages

RUN bash /tmp/setup.sh configure
COPY jupyter /home/docker/.jupyter

RUN bash /tmp/setup.sh clean

USER docker
WORKDIR /home/docker
CMD jupyter notebook
# pkill -SIGHUP dockerd
