This tool implements the WOW (Wavelet Obtained Weights) steganographic algorithm. The STC (Syndrome Trellis Codes) method is used for message embedding.

## Usage:
```
./main.py <command> [options] <image_path>
```

## Commands:
- **embed**
- **extract**
- **info** — display how many bytes can be embedded
- **cost_map** — visualize the cost map of the image
- **xy_diff** — visualize the difference between the cover and stego images when embedding a given message
- **xy_diff_full** — visualize the difference between the cover and stego images when embedding a maximum-length message

## Options:
- **--width** — width of `H_hat` matrix used by STC algorithm. It determines how much data can be embedded. The payload can be calculated as `1 / width`. It's recommended to use the largest possible width to utilize most of the image and minimize distortion
- **--height** — height of `H_hat`. A larger value gives better results, but note that both space and time complexities are `O(2^h * n)`. If you use large height values with big images (especially with RGB(A)), you'll run out of RAM
- **--cmap** — color map for matplotlib.pyplot. Used by **cost_map**, **xy_diff**, **xy_diff_full**
- **--out-path** — output image path. If not specified, the image is written to stdout. Used by **embed**, **extract**, **cost_map_luma**