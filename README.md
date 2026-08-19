# HSI Classification Baseline

## Codebase Acknowledgment

The baseline network (`models/baseline_net.py`) is developed based on the open-source 
3D CNN framework from the **AsyFFNet** project (Asymmetric Feature Fusion Network for HSI Classification).
We restructured the feature extraction pipeline by:
- Introducing multi-scale 3D dense connection units (Unit blocks with kernel sizes 3, 5, 7).
- Integrating a tri-branch attention mechanism (spectral-spatial-channel).

All contrastive methods under `compare/` (e.g., Lite-HCNet) are included for fair 
evaluation and will be properly cited in our manuscript.

## Environment
- Python 3.x
- PyTorch &gt;= 1.10
- GDAL, scikit-learn, scipy