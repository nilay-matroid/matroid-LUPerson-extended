# matroid-LUPerson-extended

## Overview
Final appearance search LUPerson code to be ported to Matroid

## Environment

**Conda**
Set up a seperate conda environment first for isolation and clean installation.
```bash
conda create --name matroid-serving python=3.7
```


**Requirements** \
You need to install these requirements to run the scripts successfully.
```bash
cd environment
pip install -r serving_requirements.txt
pip install -r extra_requirements.txt
```

**Note:** \
The following file contains packages installed in serving but it is not possible to install them simply without conflicts.
However, these packages aren't necessary for running scripts in this repository.
If you wish to see them, run
```bash
cat incompatible_serving_packages.txt
```