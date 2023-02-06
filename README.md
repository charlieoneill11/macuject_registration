# Retinal image registration using deep learning
2D non-rigid registration using self-supervised deep learning on retinal fundus images.

## Ideas
Later images have worse registration because geographic atrophy has worsened. How can we prevent this?
* Could use semantic segmentation to block out the atrophy, then train the model, then unmask and register.
* Could register each image to the previous image? Sort of a recurrent process.
