# CausalForce

Official implementation of **CausalForce: Disentangled Causal Inference for Conformal Risk Tube Prediction**.

## ⚙️ Getting Started

### 📝 System Setup
* **Operating System:** Linux Ubuntu
* **Python Version:** 3.7+
* **PyTorch Version:** 1.10.1
* **CUDA Version:** 11.3

### 📥 Dependency Installation

1. Clone the Repository
    ```bash
    git clone https://github.com/waanney/CausalForce.git
    cd CausalForce
    ```

2. Create and activate Conda environment:
    ```bash
    conda env create -f environment.yml --name CausalForce
    conda activate CausalForce
    ```

### 📦 Datasets Downloads

We use the **Multiple Coexisting Risks Dataset**, which integrates four risk categories: Interaction, Collision, Obstacle, and Occlusion.

* Download `Multiple_Coexisting_Risks_Dataset` [here](https://drive.google.com/drive/folders/13hRzEaJadxPIgf_hRIaQVJq9Pr1SsGIL?usp=drive_link). Extract all `train{xx}.zip` files into the same dataset folder.
* Please refer to the [dataset description](./Multiple_Coexisting_Risks_Dataset_Description/README.md) for details.

---

## 🚀 Usage

Navigate to `CausalForce`:
```bash
cd CausalForce
```

### Stage 1: Risk Category Classifier Pre-training
```bash
python train_cls.py
```

### Stage 2: CausalForce Training
```bash
python train_causal.py
```

### Evaluation
```bash
# CausalForce Evaluation
python inference_causal.py --checkpoint /path/to/checkpoint.ckpt

# Baseline Evaluation
python inference.py               # Given GT Bounding Box
python inference_yolo_detector.py # Given Perception Bounding Box
```

### Visualization & Downstream Task (Braking Alerts)
```bash
python vis_roi_and_save_braking_alerts.py --mode 'vis_save'
python vis_roi_and_save_braking_alerts.py --mode 'metric'
```

---

## 🙌 Acknowledgment

We acknowledge that implementations used in this project are adapted from RiskBench and SAOCP.
