docker build docker -t pointillism

sudo docker run -it --network=host --gpus all -v $PWD:/pointillism  pointillism bash
