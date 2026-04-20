# day 1

#### 环境说明

```
主环境为wsl2 ubuntu20.04 
workdir: /root/training-framework-roadmap-lab
```

```
nvidia-smi
```

#### 搭环境

```
sudo apt install -y git curl wget build-essential python3 python3-venv python3-pip tree
```

##### python环境

```
python版本：3.10

sudo apt update

sudo apt install -y \
  build-essential wget curl \
  zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
  libssl-dev libreadline-dev libffi-dev libsqlite3-dev \
  libbz2-dev liblzma-dev tk-dev uuid-dev

cd /usr/src
sudo wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
sudo tar -xzf Python-3.10.14.tgz
cd Python-3.10.14

sudo ./configure --enable-optimizations
sudo make -j"$(nproc)"
sudo make altinstall

安装报错：

Q1：Could not build the ssl module! Python requires a OpenSSL 1.1.1 or newer

A1：
1、先装openssl
sudo apt update
sudo apt install -y libssl-dev pkg-config

openssl version
pkg-config --modversion openssl || pkg-config --modversion libssl

2、重新编译安装
cd /usr/src/Python-3.10.14
make distclean 2>/dev/null || true

./configure --enable-optimizations --with-openssl=/usr --with-openssl-rpath=auto
make -j"$(nproc)"
sudo make altinstall

Q2: The necessary bits to build these optional modules were not found: _dbm To find the necessary bits, look in setup.py in detect_modules() for the module's name

A2: apt install -y libgdbm-dev libgdbm-compat-dev


```
