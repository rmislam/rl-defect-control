# Deep RL Control of Active Nematic Defects

C implementation of 2D Beris-Edwards nematodynamics with activity patterns, along with deep reinforcement learning (RL) controller training, testing, and visualization, for the paper "Controlling Defects and Probing Dynamics for Active Nematics with Deep Reinforcement Learning" by Russ Islam, Kyogo Kawaguchi, and Yuto Ashida (2026).

## Environment
Using a Python virtual environment is recommending for managing dependencies for plotting (`python3 -m venv venv` from the repo root folder). After creating a `venv` environment (simply called `venv` below), run these commands to install the Python dependencies:
```
source venv/bin/activate
pip install -r requirements.txt
```

Note also that CUDA must be installed, including the compiler `nvcc`. The simulation may silently fail by outputing zero matrices for environment observations if CUDA is not installed or linked properly.

## Configuration
Modify `./src/cpp/parameters.c` with the `GEOM_NAME` of your choice. For training geometries, the values of `I` and `J` should both be set to `420`. For the test geometry (cross maze), `I` and `J` should both be `660`. The `ALLOW_MULTIPLE_DEFECTS` parameter may be set to either `"TRUE"` or `"FALSE"`.

## Saved model weights
Saved model weights may be downloaded from this [Google Drive link](https://drive.google.com/file/d/1mpO0PMJ8ANLVtdOCdcYdYGndmsobZ_05/view?usp=drive_link).

The contents of the unzipped file should be placed in the `./learning/saved_models` directory. Subfolder names and filenames should not be modified; the scripts in this repo rely on the exact filenames.

## How to run
This command will compile the simulation, run it, and produce two mp4 movies (one for the velocity field, one for the director field) visualizing the simulation output.
```
./bin/run_all.sh
```

To run the simulation without generating movies, run
```
./bin/run.sh
```

After starting either one of these scripts, the output will show that the simulation has been initialized and is waiting for control commands. In a different terminal tab, run
```
./learning/test/test_cross_junction_down.py
```
or another script in the `./learning/test` directory. Ensure that the value of `GEOM_NAME` in `./src/cpp/parameters.c` matches your script of choice before running it.

The first terminal tab (where the simulation is running) should begin printing output. These two scripts will communicate via named pipes, alternating between control and environment steps, until the test control script has reached the maximum episode length or the +1/2 defect has been annihilated.

