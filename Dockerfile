
#FROM registry.cn-shanghai.aliyuncs.com/dlight/pytorch1.1:1.1.0
# FROM ufoym/deepo:tensorflow-py36-cu90
FROM ufoym/deepo:pytorch-py36-cu90
MAINTAINER cuihao.leo@gmail.com

# RUN pip install -r requirements.txt
RUN pip install Pillow==5.3.0
RUN pip install scipy tqdm h5py
# RUN pip install pytorch==1.0.2
# RUN pip install torch torchvision
RUN pip install -i  https://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com tensorflow-gpu==1.10.0
RUN apt update && apt install -y libsm6 libxext6 libxrender1
# RUN apt-get install libsm6 libxrender1 libfontconfig1
RUN pip install -i  https://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com opencv-python

# cleverhans==3.0.0

RUN mkdir /checkpoints \
    && cd /checkpoints \
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/senet154.pth.tar \
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/densenet121.pth.tar \
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/mobilenet_v2.pth.tar \
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/resnet18.pth.tar \
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/inceptionv4.pth.tar\
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/inceptionresnetv2.pth.tar\
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/resnet34.pth.tar \
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/resnet152.pth.tar \
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/se_resnet50.pth.tar \
    && wget http://114.214.167.31:8000/bianhy/ijcai/defence/fusai/6/xception.pth.tar

RUN mkdir /results \
    && cd /results && mkdir /mine

ADD . /competition
WORKDIR /competition
