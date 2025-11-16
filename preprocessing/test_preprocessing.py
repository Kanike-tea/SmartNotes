import argparse
from pipeline import process_note

def main():
    parser = argparse.ArgumentParser(description="Test SmartNotes preprocessing pipeline")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to test image (JPG/PNG)")

    args = parser.parse_args()

    print("\n=== Running preprocessing pipeline ===\n")
    result = process_note(args.image)

    print("\n=== Final Output ===")
    print(result)

if __name__ == "__main__":
    main()
