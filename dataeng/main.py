from loguru import logger

from src.modules.pdf import pdf_rep
from src.modules.vector_database import vector_database_svc


@logger.catch
def main() -> None:
    pdfs = pdf_rep.impl.read_pdfs()
    vector_database_svc.impl.build_vector_database(pdfs)


if __name__ == "__main__":
    main()
