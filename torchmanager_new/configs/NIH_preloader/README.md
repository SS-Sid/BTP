# NIH Data Preloader

* This prelaoder loads the NIH data reads the PNG images from folders.
* It applies PIL tranforms (because they run on CPU)
* and save them after converting to tensor.

> Reason: PIL open is a major bottleneck
    >> for a batch of 64, it takes 2.5 sec. whereas model forward and backward is very fast (hundreds ms.)

## currently performing:
1. resize to 224.
2. grayscale to single channel.
3. to tensor.