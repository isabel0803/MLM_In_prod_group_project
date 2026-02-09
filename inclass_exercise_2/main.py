import argparse
import qrcode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to encode into a QR code")
    parser.add_argument("-o", "--out", default="qrcode.png", help="Output file name")
    args = parser.parse_args()

    img = qrcode.make(args.url)
    img.save(args.out)
    print(f"QR code saved as {args.out}")


if __name__ == "__main__":
    main()

