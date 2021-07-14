# GPNet-simulator: evaluating robotic grasping in simulation

This repository provides the simulation used in GPNet as a more convenient Python package.
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

Grasps are specified as a 6dof pose of position and quaternion. The simulation reads grasps from a text file with following structure:

```
shape1_name
x, y, z, rw, rx, ry, rz
x, y, z, rw, rx, ry, rz
shape2_name
x, y, z, rw, rx, ry, rz
...
```

Simulation can be used either via CLI:

```
cd GPNet-simulator
python -m gpnet_sim -t gpnet_data/prediction/nms_poses_view0.txt
```

... or via API, as in `examples/sim_main.py`:

```
import os

import gpnet_sim

project_dir = os.path.join(os.path.dirname(__file__), '..')

conf = gpnet_sim.default_conf()
conf.testFile = os.path.join(project_dir, 'gpnet_data/prediction/nms_poses_view0.txt')
conf.z_move = True

top10, top30, top50, top100 = gpnet_sim.simulate(conf)
```

See `gpnet_sim/simulator.py` for complete list of arguments.