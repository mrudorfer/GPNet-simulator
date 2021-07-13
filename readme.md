# GPNet-simulator: evaluating robotic grasping in simulation

This repository provides the simulation used in GPNet as a convenient Python package.
The original code can be found in the [GPNet repository](https://github.com/CZ-Wu/GPNet), 
which contains the Pytorch implementation of the GPNet paper:
[Grasp Proposal Networks: An End-to-End Solution for Visual Learning of Robotic Grasps](https://arxiv.org/abs/2009.12606).

The original code has only been modified to allow for better usability.

## Installation

```
cd GPNet-simulator
pip install .
```

## Usage

To test the predicted grasps in simulation environment:

```
cd GPNet-simulator
python -m gpnet_sim.simulator -t gpnet_data/prediction/nms_poses_view0.txt
```
