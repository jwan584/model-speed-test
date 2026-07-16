import sys


def main() -> None:
    a, b = map(int, sys.stdin.buffer.read().split())
    print(a + b)


if __name__ == "__main__":
    main()
