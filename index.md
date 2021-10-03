## Find The Way Back
authors: Ting-Wei Chang, Wei-Chen Chiu, Ching-Chun Huang

### Introduction
We address the task of zero-shot blind image super-resolution and propose a flow-based generative model, named as invertible kernel estimator (IKE) which aims to recover the high-resolution details from the low-resolution input image under a challenging problem setting of having no external training data, no prior assumption on the downsampling kernel, and no pre-training components used for estimating the downsampling kernel.

### Algorithm
IKENet approximates the image downsampling and upsampling processes simultaneously via a flow-based generative model. Thanks to the invertibility of IKENet, the forward and backward flows are directly linked to the downsampling/kernel-estimating and upsampling/super-resolving steps. We provide the detailed algorithms of the forward and backward processes of the proposed IKENet with the following video.
<div style="text-align:center;">
<iframe width="600" height="400" src="https://www.youtube.com/embed/r59-rO4IVmQ" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></div>

### Experiments and Results
#### DIV2k Track 2 (unknown degradation)
The following table presents the super-resolved images on DIV2K dataset Track 2. The first two and the last two rows are the results of upsampling by 2 times and 4 times respectively.
![image](https://user-images.githubusercontent.com/11616733/135750391-39cfeccc-527d-4f29-b6db-8fd5357e234a.png)

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
