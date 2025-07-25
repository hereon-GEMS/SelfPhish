# SelfPhish
--- 
#### Self-supervised physics-informed generative networks for phase retrieval from a single X-ray hologram

[![DOI](https://zenodo.org/badge/991883128.svg)](https://doi.org/10.5281/zenodo.16413563)
![SelPhish Architecture](data/images/selfphish.png)
**Abstract:** X-ray phase contrast imaging significantly improves the visualization of structures with weak or uniform absorption, broadening its applications across a wide range of scientific disciplines. Propagation-based phase contrast is particularly suitable for time- or dose-critical in vivo/in situ/operando (tomography) experiments because it requires only a single intensity measurement. However, the phase information of the wave field is lost during the measurement and must be recovered. Conventional algebraic and iterative methods often rely on specific approximations or boundary conditions that may not be met by many samples or experimental setups. In addition, they require expert tuning of the reconstruction parameters, making them less adaptable for complex or variable conditions. Here we present a self-learning approach for solving the inverse problem of phase retrieval in the near-field regime of Fresnel theory using a single intensity measurement (hologram). A physics-informed generative adversarial network is employed to reconstruct both the phase and absorbance of the unpropagated wave field in the sample plane from a single hologram. Unlike most state-of-the-art deep learning approaches for phase retrieval, our approach does not require paired, unpaired, or simulated training data. This significantly broadens the applicability of our approach, as acquiring or generating suitable training data remains a major challenge due to the wide variability in sample types and experimental configurations. The algorithm demonstrates robust and consistent performance across diverse imaging conditions and sample types, delivering quantitative, high-quality reconstructions for both simulated data and experimental datasets acquired at beamline P05 at PETRA III (DESY, Hamburg), operated by Helmholtz-Zentrum Hereon. Furthermore, it enables the simultaneous retrieval of both phase and absorption information.


![Wire Phase Retrieval GIF](data/wire/wire_learn.gif)
## 📦 Installation & Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/daveabiy/SelfPhish.git
   cd SelfPhish
   ```

2. **Create a Python environment**  
   (Example with Conda)  
   ```bash
   conda create -n selfphish python=3.12
   conda activate selfphish
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include:
   - PyTorch ≥1.12  
   - torchvision  
   - scikit-image  
   - numpy  

4. **GPU / CUDA**  
   By default the code uses `cuda:0` if available; you can override via the `device` parameter in your experiment.

---

## 🚀 Quick Test & Usage

You can quickly test SelfPhish by running the provided `qr.py` script. This script sets up the environment, runs a simulation, trains the model, and visualizes the results:

```bash
python qr.py
```

The script will automatically handle the necessary imports and path setup. You can modify `qr.py` to run your own experiments or change parameters as needed.

Enjoy experimenting with SelfPhish! 🚀