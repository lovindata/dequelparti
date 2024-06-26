from dataclasses import dataclass


@dataclass
class EnvsConf:
    output_path = "./data"


envs_conf_impl = EnvsConf()
