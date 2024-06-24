from loguru import logger


@logger.catch
def main() -> None:
    print("Hello world!")


if __name__ == "__main__":
    main()
