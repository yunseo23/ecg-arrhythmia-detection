from pathlib import Path
import json



# -----------------------------------------
# 2) 내보내기 함수
# -----------------------------------------
def export_hyperparams(cfg: dict,
                   path: str | Path = "hyperparams.json",
                   ) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


