# Experiments Directory

This directory contains Python scripts for running various experiments related to Safe Preference Learning (SPL) using different methods. Below are details about each script along with the required command line inputs.

## 1. BT_for_splits

Runs Bradley-Terry baseline method for 10 test-train splits.

### Command Line Input:
The following input runs the experiment for all participants.

```bash
python3 ./experiments/BT_for_splits.py --seed <seed_no> --scenario <scn> --all_participants
```

## 2. SVM_for_splits

Runs SVM classification baseline method for 10 test-train splits.

### Command Line Input:
The following input runs the experiment for all participants.

```bash
python3 ./experiments/SVM_for_splits.py --seed <seed_no> --scenario <scn> --all_participants
```

## 3. gradient_based_human_subject_study

Runs the SPL experiment with the gradient-based method.

### Command Line Input:
The following input runs the experiment for all participants with a given seed for a given split.

```bash
python3 ./experiments/gradient_based_human_subject_study.py --seed <seed_no> --scenario <scenario> --split <split> --all_participants
```

## 4. random_sampling_human_subject_study

Runs the SPL experiment with random sampling.

### Command Line Input:
The following input runs the experiment for all participants with a given seed for a given split.
```bash
python3 ./experiments/random_sampling_human_subject_study.py --seed <seed_no> --scenario <scenario> --split <split> --all_participants
```

### 5. safety_check_with_BT

Runs safety tests for the stop_sign scenario using Bradley-Terry.

### Command Line Input:
The following input runs the experiment.
```bash
python3 ./experiments/safety_check_with_BT.py
```

### 6. safety_check_with_SVM

Runs safety tests for the stop_sign scenario using Support Vector Machines (SVM).

Command Line Input:
```bash
python3 ./experiments/safety_check_with_SVM.py
```
