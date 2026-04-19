# L1 NetMET — ML-based MET Reconstruction for the CMS Level-1 Trigger

**L1 NetMET** is the first machine-learning regression model targeted for operational deployment in the [CMS Level-1 Trigger](https://cms.cern.ch/iCMS/analysisadmin/cadi?ancode=TRG) at CERN. It replaces a classical pile-up mitigation algorithm with an XGBoost model that regresses L1 MET to offline PUPPI MET using full calorimeter information and object sums.

> **Key results:**
> - ~16% efficiency gain at fixed trigger rate
> - Recovers ~20 GeV of trigger threshold — directly increasing physics data collection in Run-3
> - **45 ns latency** (9 pipeline cycles), fully pipelined
> - <10% FPGA resource utilization (3% FF, 9% LUT on xc7vx690t)
> - Integrated into CMSSW_15_0_3, targeting 2025 online deployment

This is a concrete example of deploying ML under hard real-world constraints: fixed hardware (a specific Xilinx FPGA on the detector), microsecond-scale latency budgets, and no ability to iterate once in production.

---

## How It Works

The model is trained to predict offline PUPPI MET (after muon subtraction) from a small set of L1 trigger features:
- `L1 MET_pt` (after pile-up mitigation)
- `L1 MHT_pt`
- Number of trigger towers (`nTT`)
- Four leading jet pT values

An XGBoost BDT was selected over DNNs after architecture comparison — it delivers equivalent performance with a smaller resource footprint and simpler firmware integration.

The trained model is exported via [Conifer](https://github.com/thesps/conifer) and synthesized using Xilinx Vitis HLS 2024.1, targeting the demux FPGA `xc7vx690t-ffg1157-3`.

---

## Results

| Metric | Value |
|---|---|
| Efficiency gain at fixed rate | ~16% |
| Trigger threshold recovered | ~20 GeV |
| Latency | 45 ns (9 cycles) |
| FPGA FF utilization | 3% |
| FPGA LUT utilization | 9% |
| Precision | `ap_fixed<24,16>` |

---

## Installation

```bash
git clone https://github.com/jaimeleonh/netmet.git
cd netmet
git clone https://gitlab.cern.ch/cms-phys-ciemat/nanoaod_base_analysis.git --branch py3 nanoaod_base_analysis/
git clone https://github.com/jaimeleonh/L1NetMET.git
git clone https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git
git clone https://github.com/nlohmann/json.git
wget https://raw.githubusercontent.com/fastmachinelearning/hls4ml-tutorial/refs/heads/main/plotting.py -O tasks/hls4ml_plotting.py

# Run every time you open a new terminal:
source setup.sh
voms-proxy-init -voms cms -valid 192:0

# Run once after installation or when adding a new task:
law index --verbose
```

---

## Usage

### DNN Training
```bash
law run MLTraining --config-name base --version test
```
See `law run MLTraining --help` for all parameters. To launch on HTCondor (using hyperparameters from [`config/hyperopt.yaml`](https://github.com/jaimeleonh/netmet/blob/main/config/hyperopt.yaml)):
```bash
law run MLTrainingWorkflow
# To run only selected branches:
law run MLTrainingWorkflow --branches 0,1,3-5
```

### DNN Validation
```bash
law run MLValidation --config-name base --version test
```
Launch on HTCondor via `MLValidationWorkflow`. To increase memory: add `--request-cpus N`.

### BDT Training & Validation
Analogous tasks: `BDTTraining` and `BDTValidation`.

### BDT FPGA Synthesis
```bash
# Must run on HTCondor:
law run BDTSynthesisWorkflow --custom-condor-tag +lxfw=true
```

---

## Tips

- **Log dumping on HTCondor:** add `--transfer-logs` to any command. On failure, the log path will be printed.
- **No GPUs at IC:** add `--custom-condor-tag requirements=has_avx` when running on HTCondor without GPU access.

---

## People

Developed at Carnegie Mellon University in collaboration with the University of Bristol and Imperial College London, as part of the CMS Level-1 Trigger upgrade program.

**Contact:** Sena Durgut — sena.durgut@cern.ch — [github.com/senadurgut](https://github.com/senadurgut)

---

## Related

- [CMS L1 Trigger TWiki](https://twiki.cern.ch/twiki/bin/view/CMS/GlobalTrigger)
- [Conifer: BDT-to-FPGA](https://github.com/thesps/conifer)
- [hls4ml](https://github.com/fastmachinelearning/hls4ml)
