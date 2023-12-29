# PyTraject: A High-Efficient Python Tool for Backtracking and Tracking Precipitation
This model uses a parallel computing framework for trajectory calculation, of which the frequency of data loading has been reduced to only once. All the calculation process in this model is divided into two types, master node, and worker node

![image](https://github.com/Hydroscience-TiejianLi/PyTraject/assets/121434694/18f12daf-fdbb-4c5c-8713-e106390fc8ba)

The script run.py sets the input starting points’ location and precipitation information and submits it to the task module. 
The task module consists of two scripts, where points_load.py filters the starting points of air trajectories based on the spatial extent, temporal extent, and precipitation conditions from run.py and submits them to manager_points.py. 
The script manager_points.py builds the process pool and the solving of each trajectory is submitted to solver.py as a sub-process. 

![image](https://github.com/Hydroscience-TiejianLi/PyTraject/assets/121434694/232be01d-8600-4839-916e-29db70440be9)
