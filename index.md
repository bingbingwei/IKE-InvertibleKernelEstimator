## Find The Way Back
authors: Ting-Wei Chang, Wei-Chen Chiu, Ching-Chun Huang
<!-- You can use the [editor on GitHub](https://github.com/bingbingwei/IKE-InvertibleKernelEstimator/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.
 -->
### Introduction
We address the task of zero-shot blind image super-resolution and propose a flow-based generative model, named as invertible kernel estimator (IKE) which aims to recover the high-resolution details from the low-resolution input image under a challenging problem setting of having no external training data, no prior assumption on the downsampling kernel, and no pre-training components used for estimating the downsampling kernel.

### Algorithm
IKE approximates the image downsampling and upsampling processes simultaneously via a flow-based generative model. Thanks to the invertibility of IKE, the forward and backward flows of our IKE are directly linked to the downsampling/kernel estimating and upsampling/super resolving steps. We provide the detailed algorithms of the forward and backward processes of the proposed IKENet with the following video.
<iframe width="560" height="315" src="https://www.youtube.com/embed/2MGEp_uzT8U" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/bingbingwei/IKE-InvertibleKernelEstimator/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
