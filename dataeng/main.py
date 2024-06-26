from loguru import logger

from src.modules.nouveau_front_populaire.services.nouveau_front_populaire_svc import (
    nouveau_front_populaire_svc_impl,
)
from src.modules.rassemblement_national.services.rassemblement_national_svc import (
    rassemblement_national_svc_impl,
)


@logger.catch
def main() -> None:
    # nouveau_front_populaire_svc_impl.extract_and_save_as_json()
    rassemblement_national_svc_impl.extract_and_save_as_json()


if __name__ == "__main__":
    main()
