<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Find The Way Back

### Introduction
We address the task of zero-shot blind image super-resolution and propose a flow-based generative model, named as invertible kernel estimator (IKE) which aims to recover the high-resolution details from the low-resolution input image under a challenging problem setting of having no external training data, no prior assumption on the downsampling kernel, and no pre-training components used for estimating the downsampling kernel.

### Algorithm
IKENet approximates the image downsampling and upsampling processes simultaneously via a flow-based generative model. Thanks to the invertibility of IKENet, the forward and backward flows are directly linked to the downsampling/kernel-estimating and upsampling/super-resolving steps. We provide the detailed algorithms of the forward and backward processes of the proposed IKENet with the following video.
<div style="text-align:center;">
<iframe width="600" height="400" src="https://www.youtube.com/embed/gsZGBzgHGaY" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></div>

### Experiments and Results
#### DIV2k Track 2 Dataset (unknown degradation)
[download link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

The following table presents the super-resolved images on DIV2K dataset Track 2. The first two and the last two rows are the results of upsampling by 2 times and 4 times respectively.
<!-- ![image](https://user-images.githubusercontent.com/11616733/135750391-39cfeccc-527d-4f29-b6db-8fd5357e234a.png) -->
<img src="https://user-images.githubusercontent.com/11616733/135750391-39cfeccc-527d-4f29-b6db-8fd5357e234a.png" width="600">

#### Non-linear Degradation Dataset
The following table presents the super-resolved images on the non-linear degradation dataset. The first to the fourth rows are the results against Median, Bilateral, Anistopic Diffusion and Random degradation kernels respectively.
<!-- ![image](https://user-images.githubusercontent.com/11616733/135753632-37837dd2-f4d1-4cb7-bc36-64b92e445d98.png) -->
<img src="https://user-images.githubusercontent.com/11616733/135753632-37837dd2-f4d1-4cb7-bc36-64b92e445d98.png" width="600">

#### RealSR Dataset <sub><sup>[download link]
[download link](https://github.com/csjcai/RealSR)

RealSR is a dataset which consists of pairs of real-world HR and LR images and these paired images are captured with different focal lengths. The following table presents the super-resolved images on the RealSR dataset. 
<!-- ![image](https://user-images.githubusercontent.com/11616733/135754927-b00ded67-1dd4-4862-8c4c-94991a8ed87b.png) -->
<img src="https://user-images.githubusercontent.com/11616733/135754927-b00ded67-1dd4-4862-8c4c-94991a8ed87b.png" width="600">

#### Ablation Study
We perform ablation study to analyze the contributions of each design in our model using DIV2k Track 2 dataset with scale factor x2, where the results are provided in the following table. Both the tri-channel coupling layers for the IKE backbone and the bicubic residual design help to generate better SR images. Moreover, the contributions of different objective functions are discussed. Among them, $L_{energy}$ solves the color shifting problem and brings significant impact to our model. In addition, $L_{CSR}^{bwd}$ and $L_{inter}$ also play important roles for the performance improvement.

| $L_{energy}$ | Tri-Channel | Bic Res | $L_{CSR}^{bwd}$ | $L_{inter}$ |   PSNR   |  SSIM   |
| :--------:   | :---------: | :-----: | :-------------: | :---------: | :------: | :-----: |
|              |             |         |                 |             | 9.375    | 0.1326  |
|      V       |             |         |                 |             | 24.333   | 0.6412  |
|      V       |      V      |         |                 |             | 24.589   | 0.6613  |
|      V       |      V      |    V    |                 |             | 25.003   | 0.6975  |
|      V       |      V      |    V    |        V        |             | 25.122   | 0.7128  |
|      V       |      V      |    V    |        V        |      V      | 25.476   | 0.7297  |


  
