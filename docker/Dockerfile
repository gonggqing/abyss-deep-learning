FROM nvidia/cuda

# Expose jupyter notebook port
EXPOSE 8888

COPY setup_permissions.sh /tmp/setup_permissions.sh
COPY install-apt-python3 /tmp/install-apt
COPY install-pip /tmp/install-pip

RUN apt-get update &&  apt-get install -y --fix-missing $(cat /tmp/install-apt)
# RUN pip install -U $(cat /tmp/install-pip)
RUN pip3 install -U $(cat /tmp/install-pip)
RUN bash /tmp/setup_permissions.sh
# RUN apt-get clean && rm -rf /tmp/* /var/tmp/*

USER docker
ENV HOME /home/docker
CMD /bin/bash
# pkill -SIGHUP dockerd
