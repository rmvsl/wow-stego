import argparse



parser = argparse.ArgumentParser()

parser.add_argument("command", type=str)
parser.add_argument("--height", type=int, default=7, help="H_hat height. Somehow affects the embedding efficiency")
parser.add_argument("--width", type=int, default=8, help="H_hat width. The less width, the more data can be embedded. Using the biggest possible width is recommended when using default embedding mode")
parser.add_argument("image_path", type=str)

args = parser.parse_args()