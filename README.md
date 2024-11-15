# L1 NetMET
## Installation

```
git clone https://github.com/jaimeleonh/netmet.git
cd netmet
git clone https://gitlab.cern.ch/cms-phys-ciemat/nanoaod_base_analysis.git --branch py3 nanoaod_base_analysis/
git clone https://github.com/jaimeleonh/L1NetMET.git
source setup.sh  # to be run every time you open a new terminal
voms-proxy-init -voms cms -valid 192:0   # to be run every time you open a new terminal
law index --verbose  # to do only after installation or including a new task

```

## Usage
### DNN training:
```
law run MLTraining --config-name base --version test
```

Several parameters can be included, see `law run MLTraining --help`.

Launching to htcondor is possible via `law run MLTrainingWorkflow`. In this case, trainings will consider the parameters stored under [config/hyperopt.yaml](https://github.com/jaimeleonh/netmet/blob/main/config/hyperopt.yaml). In case only selected trainings want to be trained, one can add to the previous command `--branches 0,1,3-5`, where the branch numbers are set in [config/hyperopt.yaml, L30](https://github.com/jaimeleonh/netmet/blob/main/config/hyperopt.yaml#L30).

### DNN validation plots
```
law run MLValidation --config-name base --version test
```

Several parameters can be included, see `law run MLValidation --help`. These parameters will be also used when training.

Similarly, launching to htcondor is possible via `law run MLValidationWorkflow`. In this case, increasing the memory available can be done by including `--request-cpus N`.

### BDT training and validation

Analogous tasks are available, named `BDTTraining` and `BDTValidation`.


## Tips and tricks

* When running the code on htcondor, logs can be dumped to a log file via --transfer-logs. If the job fails, it will point you to the file used for dumping the logs and errors. 
* **Really important:** If running on htcondor w/o GPUs at IC, need to adid to the command `--custom-condor-tag requirements=has_avx`.



