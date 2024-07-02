from loguru import logger

from src.modules.programs_extractor import programs_extractor_svc


@logger.catch
def main() -> None:
    programs_extractor_svc.impl.run()


if __name__ == "__main__":
    main()
