FROM nvidia/cuda:9.0-cudnn7-devel
# Possibly cuda:9.0-cudnn7-devel-ubuntu16.04

# Expose jupyter notebook and tensorboard ports
EXPOSE 8888
EXPOSE 7001

COPY setup.sh /tmp/setup.sh
COPY install-pip /tmp/install-pip
COPY install-apt /tmp/install-apt

RUN bash /tmp/setup.sh prerequisites

COPY ssh /home/docker/.ssh
RUN chown -R docker /home/docker/.ssh
RUN chgrp -R docker /home/docker/.ssh
RUN chmod 600 /home/docker/.ssh/*
RUN chmod 644 /home/docker/.ssh/*.pub
COPY post-install.sh /home/docker
COPY git-deployment-clone /home/docker/bin
COPY git-deployment-pull /home/docker/bin
RUN bash /tmp/setup.sh local-packages

RUN bash /tmp/setup.sh configure
RUN bash /tmp/setup.sh clean

USER docker
WORKDIR /home/docker
CMD bash
# pkill -SIGHUP dockerd
