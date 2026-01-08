Here is a **clean, professional, research-grade `README.md`** you can directly paste into your 
# M²MVT on DAAD (Multi-View + Eye-Gaze + Episodic Memory)

This repository contains a **paper-faithful PyTorch implementation** of the **M²MVT (Multi-Modal Multi-View Transformer)** architecture on the **DAAD dataset**, incorporating **multi-view driving videos**, **real eye-gaze data**, and **episodic memory** for **driving maneuver anticipation**.

The implementation closely follows **Fig. 6 of the paper**, including memory-augmented transformers for long-term temporal reasoning.

---

##  Key Features

- Multi-view video processing (Front, Left, Right, Rear, Driver views)
- Real eye-gaze modeling using gaze CSVs from DAAD (`yaw`, `pitch`)
- Separate transformers for:
  - Multi-view perception
  - Ego + gaze stream
- **Episodic memory tokens** injected into transformer encoders
- Robust data loading (handles missing or corrupt gaze files)
- End-to-end training and evaluation pipeline in PyTorch

---

##  Architecture Overview

At a high level, the system works as follows:

1. Each camera view is processed by an independent **Video Transformer**
2. Eye-gaze is processed by a **Gaze Transformer**
3. Learnable **episodic memory tokens** are prepended to modality tokens
4. Memory-augmented transformers jointly reason over:
   - current clip information
   - past contextual memory
5. CLS tokens from multi-view and ego+gaze streams are fused for final prediction

This design enables the model to anticipate driving maneuvers using **environment perception**, **driver intent**, and **temporal context**.

---

##  Repository Structure

```

.
├── datasets/           # DAAD dataset loading logic
├── models/             # Transformers, M²MVT, episodic memory modules
├── utils/              # Video, gaze loading, collate functions, tests
├── train.py            # Training script
├── config.py           # Paths, constants, class definitions
├── README.md
└── .gitignore

````

---

##  Dataset: DAAD

This implementation assumes the **DAAD dataset** is organized as follows:

- Multi-view videos stored as `.mp4`
- Eye-gaze data provided as `*_EyeGaze.zip` files containing CSVs
- Raw `.vrs` files are ignored (not required)

 **The dataset is not included in this repository.**

Please download DAAD separately and update paths in `config.py`.

---

##  Getting Started

### 1 Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install torch torchvision pandas opencv-python
````

### 2️ Configure Dataset Path

Edit `config.py`:

```python
DATA_ROOT = Path("/path/to/daad")
```

### 3️ Sanity Check (Optional)

```bash
python -m utils.test_m2mvt_memory
```

### 4️ Train the Model

```bash
python train.py
```

---

##  Notes on Episodic Memory

* Episodic memory is implemented using **learnable memory tokens**
* Memory tokens persist across clips and are updated via attention
* Memory reset policies (e.g., episode boundaries) are not enforced in the baseline
* This matches the paper’s architectural design rather than a dataset-specific protocol

---

##  Implementation Status

* [x] Multi-view video transformers
* [x] Eye-gaze transformer
* [x] Multimodal fusion
* [x] Episodic memory (paper-faithful)
* [ ] Full-scale evaluation on complete DAAD (dataset-dependent)

---

##  Citation

If you use this code for research, please cite the original paper:

```
@article{daad_m2mvt,
  title={Early Anticipation of Driving Maneuvers Using Multi-View and Multimodal Transformers},
  author={...},
  journal={...},
  year={...}
}


##  Disclaimer

This repository is intended for **research and educational purposes** only.
It is **not an official implementation** and is **not affiliated** with the DAAD authors.

---

##  Acknowledgements

* DAAD dataset authors
* PyTorch community
* Transformer research community

---

##  Contact

For questions, discussions, or improvements, feel free to open an issue or pull request.


