# DDPMAdversarialTraining

In this work, we use diffusion models to perform adversarial training on ResNet18 models against PGD attacks. We input a subset of the model parameters and clean images to the diffusion model and output the noisy, perturbed images.

Our generated data can be found at: https://drive.google.com/drive/folders/1yMAlbCfo22qNZdrBGfs4Oq9DlQUnLw3Z?usp=sharing

Our trained diffusion model and ResNet18 model can be found at: https://drive.google.com/file/d/1IhpZrmOGbqLTffD343vsrFyPTtd7KTCC/view?usp=share_link

# Project Information

David Reitano: dreitan1

KinChin Tong: kin-tong

David has worked on the data generation code (generate_train_data.py and load_data.py) and some ablation studies (pgd_tests.py). He has also worked on generating and using data from Wang et al.'s method, since they did not record results with ResNet18 (https://github.com/wzekai99/DM-Improves-AT).

Kin has worked on implementing the diffusion model with model paramters as input and on adversrially training the ResNet18 model. 
