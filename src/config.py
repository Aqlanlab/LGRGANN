from __future__ import annotations
from pathlib import Path
from typing import Any,Dict
import yaml

def load_yaml(p:str)->Dict[str,Any]:return yaml.safe_load(Path(p).read_text())
def ensure_dir(p:str)->Path:return(d:=Path(p),d.mkdir(parents=True,exist_ok=True),d)[2]
