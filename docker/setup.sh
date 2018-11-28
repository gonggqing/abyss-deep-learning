#!/bin/bash
STAGE="$1"

function info(){
    echo "[INFO]  $*" 1>&2
}

function clone_and_install(){
    local src_dir="$1"
    local repo_uri="$2"
    local repo_name=$(basename "$repo_uri" | sed 's/\.git//')
    local ssh_key="$3"

    if [ ! -z $ssh_key ]; then
        local ssh_key="/home/docker/.ssh/git-$repo_name"
        su docker -c "mkdir -p $src_dir && cd $src_dir && /home/docker/bin/git-clone-with-key $repo_uri $ssh_key"
    else
        su docker -c "mkdir -p $src_dir && cd $src_dir && git clone $repo_uri"
    fi
    su docker -c "pip3 install --user $src_dir/$repo_name"
}

function install_prerequisites(){
    info Install system packages
    apt-get update && apt-get install -y --fix-missing $(cat /tmp/install-apt)
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
    if grep -q X11UseLocalhost /etc/ssh/sshd_config ; then
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
    chown docker /home/docker/.ssh/known_hosts
    chgrp docker /home/docker/.ssh/known_hosts
    clone_and_install "$home/src/abyss" "git@github.com:abyss-solutions/deep-learning.git" "/home/docker/.ssh/git-deep-learning"
    clone_and_install "$home/src/abyss" "git@github.com:abyss-solutions/abyss_python.git" "/home/docker/.ssh/git-abyss_python"
    clone_and_install "$home/src/abyss" "git@github.com:abyss-solutions/crfasrnn_keras.git" "/home/docker/.ssh/git-crfasrnn_keras"
    clone_and_install "$home/src" "https://github.com/matterport/Mask_RCNN.git"

    # Custom clone and install for pycocoapi
    su docker -c "git clone 'https://github.com/cocodataset/cocoapi.git' '$home/src/cocoapi' && cd '$home/src/cocoapi/PythonAPI' \
            && sed -r -i 's/python\s/python3 /g' Makefile && make"
    (cd $home/src/cocoapi/PythonAPI && make install)

    info Install pip local packages finished
    exit 0
}

function install_configure(){
    local jupyter_password=$(python3 -c "from notebook.auth import passwd; print(passwd('123'))")
    local jupyter_port=8888

    info install configure
    su docker -c 'jupyter notebook --generate-config'
    sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/" /home/docker/.jupyter/jupyter_notebook_config.py
    sed -i "s/#c.NotebookApp.password = ''/c.NotebookApp.password = '$jupyter_password'/" /home/docker/.jupyter/jupyter_notebook_config.py
    sed -i "s/#c.NotebookApp.port = 8888/c.NotebookApp.port = $jupyter_port/" /home/docker/.jupyter/jupyter_notebook_config.py
    info install configure finished
    exit 0
}

function install_clean(){
    info install clean
    # remove tensorflow so tensorflow-gpu is used
    pip3 uninstall tensorflow

    apt-get clean && rm -rf /tmp/* /var/tmp/*
    info install clean finished
    exit 0
}


[[ $STAGE == "prerequisites" ]] && install_prerequisites
[[ $STAGE == "local-packages" ]] && install_local_packages
[[ $STAGE == "configure" ]] && install_configure
[[ $STAGE == "clean" ]] && install_clean


info To run abyss_deep_learning with GPU run:

exit 0
