# PBL3 - DQN Traffic Light Control (SUMO/TraCI)

Du an nay la ban **standalone** de huan luyen dieu khien den giao thong bang DQN tren SUMO/TraCI theo phong cach paper, **nhung giu 2 actions** theo TLS thuc te cua nga tu Da Nang. Muc tieu la kiem soat dung pha den, train DQN, va bao cao ket qua theo form giong paper (cumulative negative wait time + cumulative queue size + paired analysis).

## 1) Tong quan kien truc

```
PBL3/
  pbl3_paper/
    env_sumo_tl.py          # Env SUMO/TraCI: state, action, reward, step logic
    train_dqn.py            # Train DQN + log + plot paper-style
    eval.py                 # Evaluate fixed/heuristic/DQN + histogram + t-test
    baseline_controllers.py # Fixed-time + heuristic max-queue
    aggregate_training.py   # (Optional) trung binh 3 runs nhu paper
    requirements.txt
  scenario/
    danang_16069441_10820969/
      project.sumocfg       # SUMO config
      project.net.xml       # Net
      routes_paper_weibull_5400/   # Route files per seed (eval)
  tools/
    inspect_tls_and_lanes.py       # Kiem tra TLS program + incoming lanes
    smoke_test_phase_control.py    # Force phase de xem queue/wait
    baseline_fixedtime_log.py      # Baseline fixed-time log (khong RL)
    gen_routes_weibull_paper.py    # Tao route Weibull per-seed
```

## 2) Pipeline chay chinh

### 2.1 Kiem tra TLS + lanes (khong RL)

```powershell
python .\tools\inspect_tls_and_lanes.py --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg --tls-id GS_420249146 --gui 0
```

### 2.2 Smoke test dieu khien pha

Force pha 0 -> 2 -> 0 -> 2 (10s/pha), log queue/wait tung 5s.

```powershell
python .\tools\smoke_test_phase_control.py --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg --tls-id GS_420249146 --seed 0 --log-every 5
```

### 2.3 Baseline fixed-time log (khong override TLS)

```powershell
python .\tools\baseline_fixedtime_log.py --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg --tls-id GS_420249146 --seed 0 --duration 6000 --log-every 5 --out .\runs\baseline_fixed_seed0.csv
```

### 2.4 Generate route cho eval (seeds 0..19)

```powershell
python .\tools\gen_routes_weibull_paper.py `
  --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg `
  --tls-id GS_420249146 `
  --outdir .\scenario\danang_16069441_10820969\routes_paper_weibull_5400 `
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 `
  --vehicles 1000 `
  --end 5400
```

## 3) Cau hinh DQN (paper-style, 2 actions)

### 3.1 Action space

- Dung **2 actions** tu TLS program thuc te:
  - action 0 -> green phase 0
  - action 1 -> green phase 2
- Yellow phase duoc chen tu dong (0->1, 2->3).
- `green=33s`, `yellow=6s` (giu theo TLS thuc te, khong doi 10/4 nhu paper).

### 3.2 State (80-cell binary)

- Lay incoming edges (4 huong) cua junction.
- Moi edge chia 2 lane-group:
  - group 0: xe re trai (duoc xac dinh bang route va dir_code)
  - group 1: xe di thang/phai
- Moi group chia 10 cell theo khoang cach den nut.
- Tong state dim: 4 edges x 2 groups x 10 cells = 80.

### 3.3 Reward

- `reward = 0.9 * w_prev - w_now`
- `w` la cumulative waiting time cua xe o incoming edges.

### 3.4 DQN core

- Replay buffer, epsilon-greedy, target network.
- Network: 2 hidden layers (400 units, ReLU), output linear Q-values.
- Batch size, gamma, learning rate co the tuy chinh trong CLI.

## 4) Train DQN

Train se tu generate routes moi episode (seed = `train_seed_start + episode`).

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

Output chinh:
- `runs\paper-dqn-*\models\dqn_final.keras`
- `runs\paper-dqn-*\train_log.csv`
- `runs\paper-dqn-*\plots\training_paper.png`

## 5) Evaluation (fixed vs heuristic vs DQN)

```powershell
python .\pbl3_paper\eval.py `
  --sumocfg .\scenario\danang_16069441_10820969\project.sumocfg `
  --tls-id GS_420249146 `
  --routes-dir .\scenario\danang_16069441_10820969\routes_paper_weibull_5400 `
  --model .\runs\<RUN_PAPER_TRAIN>\models\dqn_final.keras `
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```

Output chinh:
- `runs\paper-eval-*\summary.csv`
- `runs\paper-eval-*\plots\paper_eval.png`
- `runs\paper-eval-*\plots\paper_paired_diffs.png`
- `runs\paper-eval-*\paired_analysis.txt`

## 6) (Tuy chon) Average 3 training runs nhu paper

```powershell
python .\pbl3_paper\aggregate_training.py `
  --logs .\runs\paper-dqn-twoaction-run1\train_log.csv .\runs\paper-dqn-twoaction-run2\train_log.csv .\runs\paper-dqn-twoaction-run3\train_log.csv `
  --outdir .\runs\paper-dqn-twoaction-avg
```

## 7) Luu y va loi thuong gap

- **TLS ID sai**: dung `inspect_tls_and_lanes.py` de list TLS ids.
- **Khong co incoming lanes**: TLS id khong khop hoac junction khong co controlled lanes.
- **DQN khong tot**: 2 actions co the han che kha nang hoc. Tang episodes va seeds de on dinh.
- **Teleport warning**: giao thong ket xe nang, khong phai loi code; van co the train nhung ket qua se dao dong.
