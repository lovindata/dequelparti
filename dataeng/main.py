from loguru import logger

from src.confs import envs_conf
from src.modules.file_system import file_system_svc


@logger.catch
def main() -> None:
    pdfs = file_system_svc.impl.read_pdfs(envs_conf.impl.input_dirpath)


if __name__ == "__main__":
    main()
