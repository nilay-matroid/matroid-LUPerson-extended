# matroid-LUPerson-extended

## Overview
Final appearance search LUPerson code to be ported to Matroid

## Environment

**Conda** \
Set up a seperate conda environment first for isolation and clean installation.
```bash
conda create --name matroid-serving python=3.7
conda activate matroid-serving
```


**Requirements** \
You need to install these requirements to run the scripts successfully.
```bash
cd environment
pip install -r serving_requirements.txt
pip install -r extra_requirements.txt
conda install -c conda-forge/label/gcc7 rapidjson
conda install cudatoolkit=10.2 -c hcc
```

**Note:** \
The following file contains packages installed in serving but it is not possible to install them simply without conflicts.
However, these packages aren't necessary for running scripts in this repository.
If you wish to see them, run
```bash
cat incompatible_serving_packages.txt
```

## Building Triton Inference Server Python Backend

**GCC and G++ versions** \
You need to have GCC and G++ versions >=7 and <=8. \
Check the versions using the command below.
```bash
g++ --version
gcc --version
```

To install g++-7 and gcc-7.
```bash
sudo add-apt-repository ppa:jonathonf/gcc
sudo apt-get update
sudo apt install gcc-7 g++-7
```

List all existing versions of gcc and g++.
```bash
dpkg -l | grep gcc | awk '{print $2}'
dpkg -l | grep g++ | awk '{print $2}'
```

Set version 7 as default version. \
x.x represents other versions of gcc or g++.
```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 50
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-x.x 10

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 50
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-x.x 10
```

**CMAKE** \
You need to install cmake version >= 3.17 to use python backend of NVIDIA Triton server. 
- Remove existing/default cmake in ubuntu
```bash
sudo apt purge --auto-remove cmake
```

- Obtain a copy of the signing key
```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
```

- Add the repository to your sources list 
    1. For Ubuntu Focal Fossa (20.04)
    ```bash
    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
    ```

    2. For Ubuntu Bionic Beaver (18.04)
    ```bash
    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
    ```

    3. For Ubuntu Xenial Xerus (16.04)
    ``` bash
    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'
    ```

- Update and install
```bash
sudo apt update
sudo apt install cmake
```

**Installing python_backend** \
We install custom version of r22.04 as our version of python is 3.7 and not 3.8
```bash
git clone https://github.com/triton-inference-server/python_backend.git
cd python_backend
git checkout r22.04
mkdir build
cd build
cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=r22.04 -DTRITON_COMMON_REPO_TAG=r22.04 -DTRITON_CORE_REPO_TAG=r22.04 -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install
```

Set python paths
```bash
echo "export PYTHONPATH=\"${PYTHONPATH}:/home/ubuntu/Nilay/PersonReIDModels/python_backend/build/install/backends/python\"" >> ~/.bashrc
source ~/.bashrc
conda activate matroid-serving
```

## Testing without docker 
Set up the matroid-serving environment as instructed before.
```bash
cd ..
gdown 1jiC3gEYdbxd7IKSU5V_4PnNwtsxe99n8
cd matroid-LUPerson-extended
python3 tests/simple_e2e_test.py
```

## Testing with docker
**Start the server** \
Make sure nvidia-docker2 is installed.
```bash
docker run --runtime=nvidia --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:22.04-py3

git clone https://github.com/nilay-matroid/matroid-LUPerson-extended.git
cd matroid-LUPerson-extended
pip install -r environment/docker_requirements.txt

mkdir luperson_inference/1
cp luperson_inference/model.py luperson_inference/1/ 
cp luperson_inference/config.pbtxt luperson_inference/1/ 
cp -R configs luperson_inference

cd luperson_inference
gdown 1jiC3gEYdbxd7IKSU5V_4PnNwtsxe99n8
cd ..

mkdir models
cp -R luperson_inference  models/

tritonserver --model-repository `pwd`/models
```

**Start and test using client** 
```bash
docker run --runtime=nvidia -ti --net host nvcr.io/nvidia/tritonserver:22.04-py3-sdk /bin/bash
git clone https://github.com/nilay-matroid/matroid-LUPerson-extended.git
cd matroid-LUPerson-extended
cd luperson_inference
python3 client.py
```
**Expected Output**
```bash
PASS: luperson_inference
```