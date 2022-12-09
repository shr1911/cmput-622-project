# CMPUT-622-project - Towards Discovering Long Tail via Influence Estimation by Adding Noise

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
Deep learning algorithms are notorious for their propensity to very closely match the training data, frequently fitting even outliers and incorrectly labelled data points. Such fitting necessitates learning labels from training data, a phenomena that has generated a lot of study attention but has not yet been adequately explained. A theoretical explanation for this phenomena is put out by Feldman in a recent article Feldman [2019] based on the fusion of two discoveries. First off, naturally occurring image and data distributions are (informally) known to be long-tailed, meaning they contain a sizeable portion of uncommon and out-of-the-ordinary cases. Second, when the data distribution is long-tailed, such memorising is required in a straightforward theoretical model to achieve close to the ideal generalisation error. The direct empirical support for this argument was provided in Feldman and Zhang [2020a].

In this study, we develop experiments to put this theory’s main concepts to the test. The studies call for estimation of memorization values for training instances as well as the influence of each training example on accuracy at each test example after adding the Gaussian noise and Salt and Pepper noise. It would be computationally impossible to estimate these values directly, but we demonstrate that the influence and memorization values can be estimated considerably more effectively using closely similar subsampled data. Our tests show that memorising has a considerable impact on generalisation across a range of accepted metrics. They also offer quantitative and visually striking proof in support of the notion advanced in Feldman [2019]

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



