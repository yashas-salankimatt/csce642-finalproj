# Implementing DDPG for Simple Water Pouring #

---

First, ensure that all necessary dependencies are installed. 
This should encompass little more than what is required of the course assignments environment.
The only major additions should be Mujoco itself, the mujoco Python library, and the gymnasium Python library.
For simplicity, a requirements file was generated using pipreqs that may aid the installation process.

---

In order to run the code and view the simulation, use commands in the following format:

python run.py -s ddpg -t 1000 -d 'gym_examples/SimpleCup-v0' -e 150 -a 0.001 -g 0.99 -l [256,256] -b 100 --no-plots

To run the command without the simulation, and view the graph instead, use:
python run.py -s ddpg -t 1000 -d 'gym_examples/SimpleCup-v0' -e 150 -a 0.001 -g 0.99 -l [256,256] -b 100

---

The code may also be run by utilizing the "run.cmd" executable file