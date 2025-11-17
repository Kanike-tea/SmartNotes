import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from preprocessing.pipeline import process_note
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()

print("Running preprocessing pipeline...\n")
output = process_note(args.image)

print("\n===== OUTPUT =====")
for k, v in output.items():
    print(f"{k}: {v}")
