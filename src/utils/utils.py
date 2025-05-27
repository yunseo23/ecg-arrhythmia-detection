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


def print_s_distribution(train, val, test):
    for name, data in zip(['Train', 'Validation', 'Test'], [train, val, test]):
        total = len(data['y'])
        s_count = (data['y'] == 1).sum()
        non_s_count = (data['y'] == 0).sum()
     
