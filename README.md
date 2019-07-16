# tf-GMM ##
**Geometric Matching Module** in 'Toward Characteristic-Preserving Image-based Virtual Try-On Network(CP-VTON), is being implemented using Tensorflow.

## Issue ##
How to transform *torch.nn.functional.grid_sample* pytorch to tensorflow?

### grid_sample ###
https://tutorials.pytorch.kr/intermediate/spatial_transformer_tutorial.html

This function is often used in building **Spatial Transformer Networks**. STN is a generalization of differentiable attentions to any space variation. The STN allows the neural network to learn how to perform input image space transformations to enhance the geometric invariance of the model. For example, you can crop, resize, and modify the area of interest in an image. This is a very useful mechanism because CNN is not invariant to rotation, size, and more general affine deformation.

<p align="center">
<img src="./src/spatial-transformer-structure.png" width="70%" height="70%"></p>

*One of the best things about STN is that it can be easily connected to existing CNNs with little modification.*

* The localization network: takes the original image as an input and outputs the parameters of the transformation we want to apply.
* The grid generator: generates a grid of coordinates in the input image corresponding to each pixel from the output image.
* The sampler: generates the output image using the grid given by the grid generator.
<br><br>

## Usage ##
~~~
└── tf-gmm
    ├── data
        ├── train_pairs.txt
        └── train
            ├── image
            ├── pose
            ├── segment
            └── product_image
	
~~~


## Reference ##
### CP-VTON ###
[1] https://github.com/sergeywong/cp-vton

### VITON ###
[1] https://github.com/xthan/VITON

### spatial_transformer ###
[1] http://torch.ch/blog/2015/09/07/spatial_transformers.html

[2] https://pytorch.org/docs/stable/nn.html

[3] Koreans: https://tutorials.pytorch.kr/intermediate/spatial_transformer_tutorial.html



