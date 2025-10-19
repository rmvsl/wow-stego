import argparse



parser = argparse.ArgumentParser()

parser.add_argument("command", type=str)
parser.add_argument("--height", type=int, default=7, help="H_hat height. Somehow affects the embedding efficiency")
parser.add_argument("--width", type=int, default=8, help="H_hat width. Smaller width = higher embedding capacity. For the default embedding mode, it's recommended to use the largest possible width")
parser.add_argument("--cmap", type=str, default='gray', help="Color map for matplotlib.pyplot")
parser.add_argument("--save", action='store_true', help="Save images instead of showing them with matplotlib.pyplot (cost_map, xy_diff, xy_diff_full)")
parser.add_argument("image_path", type=str)

args = parser.parse_args()