# Conditional DDPM for Adversarial Training on CIFAR-10

---

## 🧠 Project Goal

This project builds a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** that generates adversarial images conditioned on the current parameters of a ResNet20 classifier.

The trained Conditional DDPM is then used to **generate model-specific adversarial images** to **adversarially train a ResNet20**, improving its robustness.

---

## 📋 Project Plan

### Step 1: Train a Conditional DDPM

- **Input:** Clean CIFAR-10 image + ResNet20 parameter embedding
- **Output:** Predict noise for DDPM training
- **Training:** Standard DDPM noise prediction loss (MSE)
- **Conditioning:** Use FiLM (Feature-wise Linear Modulation) based on ResNet parameters

### Step 2: Adversarially Train ResNet20

- For each training batch:
  - **Input:** Clean CIFAR-10 image
  - **Condition:** Current ResNet20 parameters
  - **Generate:** Adversarial image from Conditional DDPM
  - **Train:** ResNet20 on generated adversarial images
- ResNet embedding is updated dynamically as the model trains.

### Implementation Details
Our conditional diffusion model based on the original GaussianDiffusion implementation from the denoising_diffusion_pytorch library.

In the original design, the model predicts the noise added to a clean image at a randomly chosen timestep, learning to reverse the forward diffusion process. To adapt this for adversarial image generation, we extended the model to accept additional conditioning:
	•	Inputs: The diffusion model now takes both the noised image and the corresponding clean image as inputs, by concatenating them along the channel dimension (early fusion).
	•	Conditioning: We apply FiLM (Feature-wise Linear Modulation) conditioning, where the conditioning parameters (γ, β) are generated from the current ResNet parameters embedding. This allows the diffusion process to be aware of the state of the target classifier during adversarial sample generation.
	•	Training: We kept the original mean squared error (MSE) loss between predicted noise and true noise, matching the objective of standard DDPM.

Thus, our model learns to generate perturbed adversarial images specific to the current ResNet20 state, improving adversarial training effectiveness.


### Setup Instructions
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


#### Train the Conditional DDPM (For example)
python train_ddpm_from_images.py --data-dir ../train_data --epochs 500 --save-every 100 --save-dir checkout_ddpm_t500_ep500

python train_ddpm_from_images.py --data-dir ../train_data --epochs 500 --save-every 100 --timesteps 300 --save-dir checkout_ddpm_t300_ep500


#### Verify DDPM
python verify_ddpm.py --diffusion-path checkpoints_ddpm_t500_ep500/diffusion_epoch500.pth -save-dir checkpoints_ddpm_t500_ep500

#### Adversarial Training
python train_resnet_adv.py --diffusion-path ./checkpoints_ddpm/diffusion_epoch100.pth --epochs 20 --save-every 2


#### Evaluate Adversarial Training
python eval_resnet_adv.py --resnet-path checkpoints_ddpm_t500_ep500/diffusion_epoch500.pth 

