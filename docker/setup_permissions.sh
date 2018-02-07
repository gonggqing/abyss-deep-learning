#!/bin/bash

sed -i -r 's/^PermitRootLogin (\w+)/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i -r 's/^X11Forwarding (\w+)/X11Forwarding yes/' /etc/ssh/sshd_config
if [[ -z $(grep X11UseLocalhost /etc/ssh/sshd_config) ]]; then
  echo "X11UseLocalhost no" >> /etc/ssh/sshd_config
else
  sed -i -r 's/^X11UseLocalhost (\w+)/X11UseLocalhost no/' /etc/ssh/sshd_config
fi
groupadd -g 999 docker
useradd -m -u 999 -g 999 -s /bin/bash docker
passwd root <<EOF
root
root
EOF
passwd docker <<EOF
docker
docker
EOF
usermod -aG sudo docker
setfacl -dm u::rw,g:docker:rw,o::r /home/docker
echo "export QT_X11_NO_MITSHM=1" >> /home/docker/.profile
