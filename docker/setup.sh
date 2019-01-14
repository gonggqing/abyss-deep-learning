#!/bin/bash
STAGE="$1"
set -x
set -e

function info(){
    echo "[INFO]  $*" 1>&2
}

function su_docker(){
    su docker -c "$*"
}

function clone_and_install(){
    local src_dir="$1"
    local repo_uri="$2"
    local repo_name=$(basename "$repo_uri" | sed 's/\.git//')
    local ssh_key="$3"
    local flags=${@:4}

    if [ ! -z $ssh_key ]; then
        local ssh_key="/home/docker/.ssh/git-$repo_name"
        su_docker "mkdir -p $src_dir && cd $src_dir && /home/docker/bin/git-deployment-clone $repo_uri $ssh_key"
    else
        su_docker "mkdir -p $src_dir && cd $src_dir && git clone $repo_uri"
    fi
    if [[ ${flags} != *"--no-install"* ]] ; then
        su_docker "pip3 install --user $src_dir/$repo_name"
    fi
}

function install_prerequisites(){
    info Install system packages
    apt-get update && apt-get install -y --fix-missing $(cat /tmp/install-apt)

    ## Add user and group
    groupadd -g 999 docker
    useradd -m -u 999 -g docker -s /bin/bash docker
    usermod -aG sudo docker
    echo -en "docker\ndocker\n" | passwd root
    echo -en "docker\ndocker\n" | passwd docker
    setfacl -dm u::rwX,g:docker:rwX,o::r /home/docker
    su_docker "pip3 install --user -r /tmp/install-pip"

    su_docker 'mkdir /home/docker/python /home/docker/bin'
    
    source /home/docker/.containerrc

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
    echo "source /home/docker/.containerrc" >> /home/docker/.bashrc
    echo "source /home/docker/.containerrc" >> /home/docker/.profile
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
    clone_and_install "$home/src/abyss" "git@github.com:abyss-solutions/crfasrnn_keras.git" "/home/docker/.ssh/git-crfasrnn_keras" --no-install
    # Need to build crfasrnn_keras/src/cpp manually for now, after everything's installed
    su_docker "cd $home/src && git clone https://github.com/matterport/Mask_RCNN.git"
    su_docker "cd $home/src/Mask_RCNN && pip3 install --user $(grep -ivE "tensorflow" $home/src/Mask_RCNN/requirements.txt | xargs) && pip3 install --user --no-deps ."

    # Custom clone and install for pycocoapi
    su_docker "git clone 'https://github.com/cocodataset/cocoapi.git' '$home/src/cocoapi' && cd '$home/src/cocoapi/PythonAPI' \
            && python3 setup.py build_ext install --user"

    info Install pip local packages finished
    exit 0
}

function install_configure(){
    local jupyter_password=$(su docker -c 'python3 -c "from notebook.auth import passwd; print(passwd(\"123\"))"')
    local jupyter_port=8888
    local home=/home/docker

    info install configure
    su_docker '/home/docker/.local/bin/jupyter notebook --generate-config'
    chown docker "$home/.jupyter/jupyter_notebook_config.py"
    chgrp docker "$home/.jupyter/jupyter_notebook_config.py"
    sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '0.0.0.0'/" "$home/.jupyter/jupyter_notebook_config.py"
    sed -i "s/#c.NotebookApp.password = ''/c.NotebookApp.password = '$jupyter_password'/" "$home/.jupyter/jupyter_notebook_config.py"
    sed -i "s/#c.NotebookApp.port = 8888/c.NotebookApp.port = $jupyter_port/" "$home/.jupyter/jupyter_notebook_config.py"
    info install configure finished
    exit 0
}

function install_clean(){
    info install clean
    # remove tensorflow so tensorflow-gpu is used (Mask_RCNN installs it)
    pip3 uninstall -y tensorflow || true

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
