# Build ACTS With NERSC

``` bash
cd <work_dir>
git clone https://github.com/acts-project/acts.git
cd acts
git checkout releases # checkout a release version. v24
```

## Use Shifter container

``` bash
# Based on the acts official docker image,
# git-lfs is needed to download the open detector config
shifter --image=hrzhao076/acts:v1 /bin/bash
```
This docker image `hrzhao076/acts:v1` is built on the top of `ghcr.io/acts-project/ubuntu2004:v41` for the usage on NERSC as no sudo priviledge is granted as a shifter container.
The Dockerfile can be found at [Dockerfile_acts](./Dockerfile_acts).

To make sure you are in the docker container,
``` bash
cat /etc/lsb-release
```

the expected output is
```
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=20.04
DISTRIB_CODENAME=focal
DISTRIB_DESCRIPTION="Ubuntu 20.04.5 LTS"
```

## Enable the OpenDataDetector(ODD)

``` bash
git submodule init # register a submodule called thirdparty/OpenDataDetector
git submodule update # git clone, git lfs is needed
# One can check the file size of thirdparty/OpenDataDetector/data/odd-material-maps.root
# should be 14Mb

```

## Create a virtual env (Recommended)
``` bash
python3 -m venv venv
source venv/bin/activate
export PYTHONPATH=
# Install some dependencies to the virtual env
pip install -r Examples/Python/tests/requirements.txt
pip install -r Examples/Scripts/requirements.txt

pip install pytest --upgrade
```

## CMake

The build options can be found at [Build Options](https://acts.readthedocs.io/en/latest/getting_started.html#build-options)
``` bash
cmake -DACTS_BUILD_EXAMPLES_PYTHIA8=ON -DACTS_BUILD_PLUGIN_DD4HEP=ON -DACTS_BUILD_PLUGIN_JSON=ON -DACTS_BUILD_PLUGIN_TGEO=ON -DACTS_BUILD_EXAMPLES=ON -DACTS_BUILD_EXAMPLES_DD4HEP=ON -DACTS_BUILD_EXAMPLES_GEANT4=ON -DACTS_BUILD_INTEGRATIONTESTS=OFF -DACTS_BUILD_UNITTESTS=OFF -DACTS_BUILD_FATRAS_GEANT4=ON -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON -DACTS_BUILD_ODD=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -S . -B build/
# You can also check the cmake log here.

```

``` bash
cd build
make -j20 # Perlmutter is powerful
```

``` bash
source this_acts.sh
source python/setup.sh
export DD4hepINSTALL="/usr/local/"

```

And set it in the `~/.bashrc`
```
echo "export acts_path='/acts'" >> ~/.bashrc
echo "source \${acts_path}/venv/bin/activate" >> ~/.bashrc
echo "source \${acts_path}/build/this_acts.sh" >> ~/.bashrc
echo "source \${acts_path}/build/python/setup.sh" >> ~/.bashrc
echo "export DD4hepINSTALL='/usr/local/'" >> ~/.bashrc
```

## Run the Example

``` bash
mkdir run_examples
# check the help info
python3 ../../Examples/Scripts/Python/full_chain_odd.py -h

# Run with the default
python3 ../../Examples/Scripts/Python/full_chain_odd.py

# Run my custom script
python3 ../../Examples/Scripts/Python/full_chain_odd_PV_study.py --ttbar -n 1000 -npu 50 -nthd 10 | tee log.n1000.npu50.txt
```

The output folder is `odd_output` with the following structure.

```
odd_output/
├── estimatedparams.root
├── hits.root
├── measurements.root
├── particles_final.root
├── particles_initial.root
├── performance_ambi.root
├── performance_ckf.root
├── performance_seeding.root
├── performance_vertexing.root
├── timing.tsv
├── trackstates_ambi.root
├── trackstates_ckf.root
├── tracksummary_ambi.root
└── tracksummary_ckf.root

0 directories, 14 files

```

# Note
1. `Visutal Studio Code` build the software with `-DCMAKE_BUILD_TYPE:STRING=Debug` enabled, that means the production is very slow.
