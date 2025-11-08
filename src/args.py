import argparse



parser = argparse.ArgumentParser()

parser.add_argument("command", type=str)
parser.add_argument("--height", type=int, default=7, help="H_hat height. Somehow affects the embedding efficiency")
parser.add_argument("--width", type=int, default=8, help="H_hat width. Smaller width = higher embedding capacity. It's recommended to use the largest possible width")
parser.add_argument("--cmap", type=str, default='gray', help="Color map for matplotlib.pyplot")
parser.add_argument("--out-path", default=None, help="Output image path. If not specified, the image is written to stdout")
parser.add_argument("image_path", type=str)

args = parser.parse_args()