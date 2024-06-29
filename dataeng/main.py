from loguru import logger

from src.modules.ensemble.services import ensemble_svc
from src.modules.nouveau_front_populaire.services import nouveau_front_populaire_svc
from src.modules.rassemblement_national.services import rassemblement_national_svc


@logger.catch
def main() -> None:
    ensemble_svc.impl.extract_and_save_as_json()
    nouveau_front_populaire_svc.impl.extract_and_save_as_json()
    rassemblement_national_svc.impl.extract_and_save_as_json()


if __name__ == "__main__":
    main()
