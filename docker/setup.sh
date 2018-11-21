#!/bin/bash
STAGE="$1"

function info(){
    echo "[INFO]  $@" 1>&2
}

function clone_and_install(){
    local src_dir=$1
    local repo_name=$2
    su docker -c "mkdir -p $src_dir && cd $src_dir && /home/docker/bin/git-clone-with-key git@github.com:abyss-solutions/$repo_name.git /home/docker/.ssh/git-$repo_name"
    su docker -c "pip3 install --user $src_dir/$repo_name"
}

function install_prerequisites(){
    info Install system packages
    apt-get update && apt-get install -y --fix-missing $(cat /tmp/install-apt)
    # apt-get update && apt-get install -y --fix-missing python3-pip
    pip3 install -U pip
    pip3 install -r /tmp/install-pip

    ## Add user and group
    groupadd -g 999 docker
    useradd -m -u 999 -g docker -s /bin/bash docker
    usermod -aG sudo docker
    echo -en "docker\ndocker\n" | passwd root
    echo -en "docker\ndocker\n" | passwd docker
    setfacl -dm u::rwX,g:docker:rwX,o::r /home/docker

    # Set default permissions
    # chown -R docker /home/docker
    # chgrp -R docker /home/docker

    su docker -c 'mkdir /home/docker/python /home/docker/bin'
    echo 'export PYTHONPATH=/home/docker/python:$PYTHONPATH' >> /home/docker/.bashrc
    echo 'export PATH=/home/docker/bin:/home/docker/.local/bin:$PATH' >> /home/docker/.bashrc
    source /home/docker/.bashrc

    # Set up X11 forwarding
    sed -i -r 's/^PermitRootLogin (\w+)/PermitRootLogin yes/' /etc/ssh/sshd_config
    sed -i -r 's/^X11Forwarding (\w+)/X11Forwarding yes/' /etc/ssh/sshd_config
    if [[ -z $(grep X11UseLocalhost /etc/ssh/sshd_config) ]]; then
      echo "X11UseLocalhost no" >> /etc/ssh/sshd_config
    else
      sed -i -r 's/^X11UseLocalhost (\w+)/X11UseLocalhost no/' /etc/ssh/sshd_config
    fi

    # Some X11 bugfix
    echo "export QT_X11_NO_MITSHM=1" >> /home/docker/.profile
    info Install system packages finished
    exit 0
}

function install_local_packages(){
    info Install pip local packages
    local home=/home/docker
    # ssh-agent "bash"
    ssh-keyscan github.com >> /home/docker/.ssh/known_hosts
    clone_and_install "$home/src/abyss" "deep-learning"
    clone_and_install "$home/src/abyss" "abyss_python"
    clone_and_install "$home/src/abyss" "crfasrnn_keras"
    info Install pip local packages finished
    exit 0
}

function install_configure(){
    info install configure
    su docker -c 'jupyter notebook --generate-config'
    info install configure finished
    exit 0
}

function install_clean(){
    info install clean
    apt-get clean && rm -rf /tmp/* /var/tmp/*
    info install clean finished
    exit 0
}


[[ $STAGE == "prerequisites" ]] && install_prerequisites
[[ $STAGE == "local-packages" ]] && install_local_packages
[[ $STAGE == "configure" ]] && install_configure
[[ $STAGE == "clean" ]] && install_clean
exit 1