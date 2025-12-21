# PBL3 – DQN Traffic Light Control (SUMO/TraCI)

Thư mục này là bản **copy standalone** để làm theo hướng **paper-like**: 4 actions (NSA/NSLA/EWA/EWLA), state binary 80 dims và reward delta-wait, chạy trên junction OSM (Đà Nẵng).

## 0) Smoke test điều khiển phase + baseline logging (không RL)

1) Inspect TLS program + incoming lanes:
```powershell
python .\tools\inspect_tls_and_lanes.py --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg --tls-id GS_420249146 --gui 0
```

2) Smoke test set phase 0→2→4→2 (mỗi phase 10s) và in queue/wait:
```powershell
python .\tools\smoke_test_phase_control.py --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg --tls-id GS_420249146 --seed 0 --log-every 5
```

3) Baseline fixed-time (không override TLS) log ra CSV:
```powershell
python .\tools\baseline_fixedtime_log.py --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg --tls-id GS_420249146 --seed 0 --duration 6000 --log-every 5 --out .\runs\baseline_fixed_seed0.csv
```

## 1) Cài đặt (Windows / PowerShell)

```powershell
cd C:\Users\halem\PBL3
python -m venv .venv
.venv\Scripts\Activate
pip install -U pip
pip install -r .\pbl3_paper\requirements.txt
```

## 2) Xác định TLS ID (nếu cần)

```powershell
python .\tools\inspect_sumo_scenario.py --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg
```

## 3) Setup (2 actions / 80-cell binary / Weibull 5400s)

Mode này dùng **2 actions** theo TLS program thực tế: chọn **green phase 0 hoặc 2** (yellow tương ứng 1 và 3). State là binary **80 dims** (4 incoming edges × {left,straight/right} × 10 cells) và reward **delta-wait** `0.9*w_prev - w_now`.

### 3.1 Generate routes (eval seeds 0..19)

```powershell
python .\tools\gen_routes_weibull_paper.py `
  --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg `
  --tls-id GS_420249146 `
  --outdir .\scenario\danang_16069441_10820969\routes_paper_weibull_5400 `
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 `
  --vehicles 1000 `
  --end 5400
```

### 3.2 Train DQN

Script sẽ tự generate route mỗi episode (seed = `--train-seed-start + episode`) vào `runs\paper-dqn-*\routes\`.

```powershell
python .\pbl3_paper\train_dqn.py `
  --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg `
  --tls-id GS_420249146 `
  --episodes 100 `
  --max-steps 5400 `
  --vehicles 1000 `
  --train-seed-start 100 `
  --green 33 `
  --yellow 6
```

Output: `runs\paper-dqn-*\models\dqn_final.keras` + `train_log.csv` + plot.

### 3.3 Evaluation (fixed-time vs heuristic vs DQN)

```powershell
python .\pbl3_paper\eval.py `
  --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg `
  --tls-id GS_420249146 `
  --routes-dir .\scenario\danang_16069441_10820969\routes_paper_weibull_5400 `
  --model .\runs\<RUN_PAPER_TRAIN>\models\dqn_final.keras `
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```

Output: `runs\paper-eval-*\summary.csv` + `plots\comparison.png`.
