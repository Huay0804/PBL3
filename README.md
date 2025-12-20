# PBL3 – DQN Traffic Light Control (SUMO/TraCI)

Thư mục này là bản **copy standalone** để train + evaluate DQN cho junction OSM (Đà Nẵng) bằng chương trình đèn hiện có.

## 1) Cài đặt (Windows / PowerShell)

```powershell
cd C:\Users\halem\PBL3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r .\pbl3_rl\requirements.txt
```

## 2) Xác định TLS ID (nếu cần)

```powershell
python .\tools\inspect_sumo_scenario.py --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg
```

## 3) Train DQN

```powershell
python .\pbl3_rl\train_dqn.py `
  --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg `
  --tls-id GS_420249146 `
  --episodes 50 `
  --demand randomtrips
```

Output: `pbl3_rl\runs\dqn-*\` (CSV log + model `.keras` + plots).

## 4) Evaluation (fixed-time vs heuristic vs DQN)

1) Chạy baseline (không cần model):
```powershell
python .\pbl3_rl\eval.py `
  --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg `
  --tls-id GS_420249146 `
  --demand randomtrips `
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```

2) Chạy đủ 3 mode (thêm `--model`):
```powershell
python .\pbl3_rl\eval.py `
  --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg `
  --tls-id GS_420249146 `
  --model .\pbl3_rl\runs\<RUN_TRAIN>\models\dqn_final.keras `
  --demand randomtrips `
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```

Output: `pbl3_rl\runs\eval-*\summary.csv` + plots trong `plots\`.

