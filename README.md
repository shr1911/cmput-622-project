# Cmput-622-project - Towards Discovering Long Tail via Influence Estimation by Adding Noise

## Team
|Student name| CCID |
|------------|------|
|Shraddha Mukesh Makwana     |    smakwana  |
|Pranjal Dilip Naringrekar   |    naringre  |
|Ashima Anand                |    ashima2  |


This git repository has been made for maintaining the project work of CMPUT 622 project on Towards Discovering Long Tail via Influence
Estimation by Adding Noise. The code here contains adding noise to mnist dataset/

Type of noise we tried :-
1. Gaussian Noise
2. Salt and pepper Noise


## List of items in README
This file contains
- [ ] Summary of project
- [ ] Project Structure
- [ ] Execution Instructions to reproduce the result 
- [ ] Acknowledgement for all resources consulted (discussions, texts, urls, etc.) while working on the project. 

## Summary of Project

## Structure
cmput-622-project folder :

- `code` 
    - `mnist-without-noise`
    - `mnist_with_gaussian_noise`
    - `mnist_with_salt_and_pepper`
    - Files for CIFAR10 dataset
- `demo_images`
- `report`: cotnains final paper

## Execution Instructions to reproduce the result

### 1. Pre-requisite setup to run the code
- Need high computing resource for CIFAR10. For small data can run in jupyter notebooks
- We tried in bonanza as well

### 2. Running notebooks
- In order to see the output, each notebooks can be ran. It will generate one pdf which contains output


## Acknowledgment and Resources
- We would like to thank Prof. Nidhi Hegde for her guidance thrughout the coourse of CMPUT-622.
- Gaussian Noise - https://www.kaggle.com/code/chanduanilkumar/adding-and-removing-image-noise-in-python/notebook
- Salt and Pepper Noise - https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/
- Base paper 1 - Vitaly Feldman. Does learning require memorization? a short tale about a long tail, 2019. URL https://arxiv.org/abs/1906.05271.
- Base paper 2 - Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation, 2020. URL https://arxiv.org/abs/2008.03703 (We have used repo of this paper for reproducing the work without noise)
- Libraries for image processing - https://docs.opencv.org/4.x/



