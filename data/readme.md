# Data Directory

The `data` directory contains essential datasets used in the experiments. This directory is organized into two subfolders: `preferences` and `trajectories`.

## 1. Preferences

The `preferences` directory includes preference data obtained from participants in the human subject study. It is further divided into two scenarios: `stop_sign` and `pedestrian`.

Please note that the human subject study was conducted under the IRB Protocol No. HUM00221976.

## 2. Trajectories

The `trajectories` directory contains trajectory data for both scenarios: `stop_sign` and `pedestrian`. We also include a violating trajectory file.

These trajectory files are provided in pickle format. You can load them using the following command in your Python code:


```python
import pickle
with open(filename, "rb") as f:
    s_tracks = pickle.load(f)


Feel free to explore the datasets and leverage them for your experiments or analyses.
