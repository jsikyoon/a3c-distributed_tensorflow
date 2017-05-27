a3c-distributed_tensorflow
===========

Distributed Tensorflow Implementation of a3c from Google Deepmind.

Implementation is on Tensorflow 1.0

http://arxiv.org/abs/1602.01783

"We present asynchronous variants of four standard reinforcement learning algorithms and show that parallel actor-learners have a stabilizing effect on training allowing all four methods to successfully train neural network controllers." Form Paper

The implementation is based on miyosuda git. https://github.com/miyosuda/async_deep_reinforce

The core implementation is almost same to miyosuda ones. The parts for running with Distributed Tensorflow are changed. 

Atari Game
-------------------

`
./auto_run.sh 
`

You need to set your hostname and port number in a3c.py code. The number of parameter servers and workers can be set in auto_run.sh script file.


### Settings
Almost Setting values are same to miyosuda ones except the number of workers. The Number of worker is 14, and the number of parameter server is 7.

### Results

![alt tag](https://github.com/jaesik817/a3c-distributed_tensorflow/blob/master/figures/a3c_dist_tensor.PNG)

The experiment is Pong game with the number of worker as above ones. 
This one runs with just CPU. The CPU and memory of server are Xeon E7-8880 v4 x 4 (176 threaing with hyper-threading) and 1TB respectively (cpu and memory are not fully used. About 50% of CPU and 20GB Memory are used).
As above result, about 800~900 steps run per seconds (that is better than 980ti performance in miyosuda git), and score is saturated at about 35M steps.
