# Real-Time Brain-to-Brain Synchrony Estimation Algorithm

**Implementation of a Real-Time Brain-to-Brain Synchrony Estimation Algorithm for Neuroeducation Applications**  
*Mendoza-Armenta et al., Sensors (2024), 24(6): 1776* :contentReference[oaicite:0]{index=0}

---

##  Overview

This repository hosts a Python-based real-time algorithm that estimates brain-to-brain (B2B) synchrony between interacting individuals using electroencephalography (EEG) signals—particularly suited for educational and collaborative contexts. The algorithm employs bispectrum analysis alongside multiprocessing to process EEG data in real time, enabling neural synchrony monitoring during social interactions such as collaborative tasks or competitive games. :contentReference[oaicite:1]{index=1}

---

## Features

- Real-time EEG preprocessing: includes filtering and artifact handling via Python.
- Bispectrum-based synchrony estimation: efficient inter-brain coupling metric using bispectrum analysis.
- Multiprocessing support: enables concurrent real-time processing of signals from two EEG sources.

Application contexts:
- Collaborative tasks (e.g., puzzle solving)—demonstrated to yield significantly higher B2B synchrony.
- Competitive tasks (e.g., one-on-one games)—provide a meaningful comparison benchmark.
- Statistical validation: determines differences using a Wilcoxon rank-sum test; 33.75% of comparisons achieved statistical significance.
- Versatile use cases: designed for neuroeducation, but easily adaptable to classrooms, industry, and varied EEG hardware setups.

---

## Repository Structure

├── brain2brain_sync/        # Python module containing core logic  
│   ├── __init__.py  
│   ├── EEG_device.py        # Connection, filtering, and EEG time-window handling  
│   ├── bispectrum.py        # Bispectrum extraction logic  
│   ├── graphs.py            # Real-time data graphs  
│   └── stopwatch.py         # Counter for master control  
├── experimental_results     # Folder created during runtime execution to store data (local)  
├── tests/  
│   └── test_bispectrum.py   # Unit testing module  
├── config.json              # File containing editable parameters to configure the experiment  
├── run_RT_B2B_v3.py         # Main script to run the experiment using multiprocessing  
├── requirements.txt         # Dependencies for reproducibility  
├── environment.yml          # (Optional) Conda environment specification  
├── LICENSE                  # CC (Creative Commons)  
├── README.md                # This file  
├── CITATION.cff             # Citation file for GitHub integration  
└── .gitignore

---

## Installation

1. git clone https://github.com/Amisaday74/REAL-TIME-B2B-.git  
2. cd REAL-TIME-B2B-

- Setting up an environment on macOS or Linux:
> python3 -m venv venv

> source venv/bin/activate

> pip install -r requirements.txt

- Setting up an environment on Windows:
> python -m venv venv

> .\venv\Scripts\activate

> pip install -r requirements.txt

- Optionally, if using Conda:
> conda env create -f environment.yml

> conda activate b2b_synchrony_env

---

## Usage


In `config.json` you will find the variables required to adapt the execution of the algorithm to your needs.

- `"board_id"`                - Write here the name of any of the available boards in BrainFlow: https://brainflow.readthedocs.io/en/stable/SupportedBoards.html
- `"test_duration_seconds"`   - Duration of the experiment in seconds. This is the total time that the algorithm will be operating.
- `"timewindow_seconds"`      - Duration in seconds of each time window. This number must be a divisor of `"test_duration_seconds"` to avoid misalignments and lost data.
- `"reference_channels"`      - A list containing all channels that must be considered as references. The script will calculate the average of all selected channels.
- `"experiment_phase"`        - Variable used to select the execution mode: either calibration or interaction.
- `"devices"`                 - A dictionary containing relevant data for both devices intended to be connected.

# Example: Calibration mode

If it is your first time running the algorithm, you must first execute a session with `"experiment_phase"` set to `"calibration"`. This will create the baseline database required for future recordings.

In this example, `config.json` is configured to extract data from the ENOPHONES every four seconds for one minute in `"calibration"` mode.

```json
{
  "board_id": "ENOPHONE_BOARD",   
  "test_duration_seconds": 60,
  "timewindow_seconds": 4,
  "reference_channels": ["CH1", "CH2"],
  "experiment_phase": "calibration",
  "devices": [
    {
      "device_id": 1,
      "device_name": "Device_1",
      "user": "User_1",
      "mac_address": "11:22:33:44:55:66"
    },
    {
      "device_id": 2,
      "device_name": "Device_2",
      "user": "User_2",
      "mac_address": "aa:bb:cc:dd:ee:ff"
    }
  ]
}
```

After saving your specific configuration, execute `"run_RT_B2B_v3.py"` to store calibration data for the first time.
Inside the script logic, every time `"run_RT_B2B_v3.py"` is executed, the user is asked for one input:

> terminal: Please write the assigned number for the dyad under analysis:  <---- Write here an integer

The experimental design of this algorithm assigns a unique number to every dyad. This keeps separate and well-organized folders for every pair of subjects inside the experimental_results directory. You can rerun the algorithm with the same "experiment_phase" and integer input to overwrite calibration results for the same dyad.

# Example: Interaction mode
Once calibration data has been stored, the script is ready to record as many experimental sessions as needed. For example, if you need to estimate brain-to-brain synchrony during a 10-minute interactive session, keep the same value for "timewindow_seconds" selected during calibration mode and set the desired recording duration in "test_duration_seconds". Then set "experiment_phase" to "interaction".

```json
{
  "board_id": "ENOHONE_BOARD",   
  "test_duration_seconds": 600,
  "timewindow_seconds": 4,
  "reference_channels": ["CH1", "CH2"],
  "experiment_phase": "interaction",
  "devices": [
    {
      "device_id": 1,
      "device_name": "Device_1",
      "user": "User_1",
      "mac_address": "11:22:33:44:55:66"
    },
    {
      "device_id": 2,
      "device_name": "Device_2",
      "user": "User_2",
      "mac_address": "aa:bb:cc:dd:ee:ff"
    }
  ]
}
```

Execute `"run_RT_B2B_v3.py"` to start the analysis as many times as needed. This time, the script will ask for two inputs at the beginning:

> terminal: Please write the assigned number for the dyad under analysis:  <---- Write here an integer

> terminal: Enter the iteration number of the current experimental test:  <---- Write here an integer

Keep writing the same number for the dyad being analyzed in the first input. Start by writing 1 in the second input and increment this value by one every time you start a new recording session.

---

## Results and Validation

Collected data in real-time is stored in the following subfolders:

├── experimental_results/       
│   └── Dyad01/ 
│       ├── Calibration_data/        
│       └── Record01_StartDatetime  
│           ├── Bispectrum             
│           ├── Figures        
│           └── Real_time_data    

In "calibration" mode, the Bispectrum folder stores the following files:
1. Calibration_data.csv      - Main bispectrum results       
2. Nested_loops.csv          - Data arranged to calculate mean bispectrum
3. Mean.csv                  - Final mean matrix of bisprectrum results

In "interaction" mode, the Bispectrum folder stores the following files:
1. Interaction_data.csv               - Main bispectrum results       
2. Frequency_bands_bispectrum.csv     - Average normalized bisprectum per timewindow grouped by frequency bands

Data contained inside Real_time_data:
1. Device_1_raw_data                  - Raw EEG from subject 1 (samples x channels) + Timestamps column
2. Device_1_signal_processing         - Preprocessed EEG signal from subject 1 (samples x channels)
3. Device_2_raw_data                  - Raw EEG from subject 2 (samples x channels) + Timestamps column
4. Device_2_signal_processing         - Preprocessed EEG signal from subject 2 (samples x channels)

Data contained inside Figures:
- Offline plotting of Frequency_bands_bispectrum.csv

Results from the pusblished article can be found in branch "rel/B2B_algorithm_v1". 
- Collaborative (puzzle-solving) tasks consistently produced higher bispectral synchrony than competitive ones.
- A Wilcoxon rank-sum test confirmed statistical significance in ~33.75% of cases.
- These differences highlight how real-time inter-brain synchrony reflects varied social cognitive contexts.

---

## License

Licensed under Creative Commons Attribution 1.0

---

## Contributor and Contact
Authors: Axel A. Mendoza-Armenta, Paula Blanco-Téllez, Adaliz G. García-Alcántar, Ivet Ceballos-González, María A. Hernández-Mustieles, Ricardo A. Ramírez-Mendoza, Jorge de J. Lozoya-Santos, Mauricio A. Ramírez-Moreno.

Repository maintained by: Axel Mendoza – feel free to open issues or pull requests!

For questions or collaborations, reach out via GitHub or email: axelmendoza47@live.com.mx

---

##  Citation

If you use or build on this software, please cite the original paper:

```bibtex
@article{mendoza-armenta2024real,
  title={Implementation of a Real-Time Brain-to-Brain Synchrony Estimation Algorithm for Neuroeducation Applications},
  author={Mendoza-Armenta, A.A. and Blanco-Téllez, P. and García-Alcántar, A.G. and Ceballos-González, I. and Hernández-Mustieles, M.A. and Ramírez-Mendoza, R.A. and Lozoya-Santos, J.d.J. and Ramírez-Moreno, M.A.},
  journal={Sensors},
  volume={24},
  number={6},
  pages={1776},
  year={2024},
  doi={10.3390/s24061776}
}