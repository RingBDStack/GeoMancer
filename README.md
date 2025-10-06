
This is the official repository for NeurIPS 2025 Paper [Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction].


### Python environment setup

```bash
conda create -n geomancer python=3.9
conda activate geomancer
pip install -r requirments.txt
```
You should install torch=2.3.1, torch-geometric=2.6.1 and pytorch-lightning=2.4.0.

### Running the code

To run the code for graph classfication, you could use the following commands.
```bash
# The first step is to pretrain an Riemannian autoencoder.
bash GeoMancer/cfg/photo-node-encoder.sh

# Then train the diffusion, please do not forget to change your checkpoint path.
bash GeoMancer/cfg/photo-node-diffusion.sh

# If you want to run other datasets or other models, you could change the parameters in GeoMancer/cfg/photo-encoder.yaml or create your own parameters file.
```

For other tasks, we have also provided the example commands in cfg.

### Cite
Welcome to cite our work!



