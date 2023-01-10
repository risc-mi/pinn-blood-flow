# Blood Flow Modeling with Physics-Informed Neural Networks (PINNs)

## Description

* This repository contains the source codes for modeling 3D blood flows using physics-informed neural networks (PINNs) described in (Moser et al., 2022, doi: tbd).
* Four geometries (cylinder, bifurcation, and two cerebral aneurysms) are included, with seven different PINN architectures each (i.e., seven different .py files).
* The trainings are performed within NVIDIA's Modulus framework v22.03 (https://developer.nvidia.com/modulus)  
* Reference CFD data were generated with MODSIM (Fenz et al., 2016, https://doi.org/10.22360/SpringSim.2016.MSM.010) and are available in the modsim subfolders.  




## Installation of NVIDIA Modulus via Docker
The entire Modulus installation process is documented on the NVIDIA Homepage (https://docs.nvidia.com/deeplearning/modulus/user_guide/getting_started/installation.html)
* ensure that Docker Engine is installed
* ensure that NVIDIA docker toolkit is installed 
```
sudo apt-get install nvidia-docker2
```
* download the Modulus docker container (version 22.03 from the NVIDIA Homepage (https://developer.nvidia.com/modulus)
* load the docker container
```
docker load -i modulus_image_v22.03.tar.gz
```
## Running a PINN training

* copy the "examples" folder to your working directory (the "examples" folder will get mounted into the docker instance)
* start the docker container
```
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
           --runtime nvidia -v ${PWD}/examples:/examples \
           -it --rm modulus:22.03 bash
```
* inside the docker instance, go to the desired simulation folder
* run .py file(s), for example:
```
python fully.py
```
* the results will be stored in the outputs folder

## Citation
Moser, P.; Fenz, W.; Thumfart, S.; Ganitzer, I.; Giretzlehner, M.; "Modeling of 3D Blood Flows with Physics-Informed Neural Networks: Comparison of Network Architectures", *Fluids*, **2023**, pp. TBD, doi: TBD  

## Acknowledgements
This project is financed by research subsidies granted by the government of Upper Austria within the research projects MIMAS.ai, MEDUSA (FFG grant no. 872604) and ARES (FFG grant no. 892166). RISC Software GmbH is Member of UAR (Upper Austrian Research) Innovation Network.