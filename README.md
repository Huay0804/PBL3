# PBL3 - DQN điều khiển đèn giao thông (SUMO/TraCI)

Dự án huấn luyện DQN điều khiển 1 junction TLS trong SUMO bằng state lane-based 80-cell binary.
Actions lấy trực tiếp từ TLS program thực tế và được kiểm tra hợp lệ bằng controlledLinks.
Mọi tham số chạy nằm trong `experiment_config.yaml`.

## 1) Yêu cầu cài đặt

- SUMO đã cài và có TraCI:
  - set biến môi trường `SUMO_HOME` trỏ tới thư mục SUMO, hoặc
  - thêm `SUMO/bin` vào `PATH` để gọi được `sumo` / `sumo-gui`.
- Python 3.8+ (Windows)

Cài deps:

```powershell
cd C:\Users\halem\PBL3
python -m venv .venv
.venv\Scripts\Activate
pip install -U pip
pip install -r .\pbl3_paper\requirements.txt
```

## 2) Cấu hình thí nghiệm

Tham số chạy nằm trong `experiment_config.yaml`:

- episodes=100, repeats=3, eval_sims=100
- sim_seconds=5400, vehicles=1000
- Weibull(shape=2) demand
- tỷ lệ hướng đi: straight=0.6, turning=0.25, u-turn=0.15
- depart_lane="best", depart_speed="auto" (SUMO tự chọn tốc độ xuất phát)
- eval_depart_speed="auto"
- action_phase_indices = [0, 2, 4, 6]
- timing mode (chọn 1):
  - `STRICT_MATCH_README` -> green_step=10, yellow_time=4 (ép bằng TraCI)
  - `KEEP_TLS_NATIVE` -> dùng duration native trong TLS program
- dqn.fit_verbose=1 (bật log Keras fit)

Nếu thay đổi config, hãy train/eval lại.

## 3) State, actions, metrics

State (80 binary cells):
- 4 arms, mỗi arm có 2 lane-group:
  - TR group: Through + Through+Right
  - LU group: Left + U-turn
- Mỗi group chia 10 cells theo khoảng cách đến junction.
- Thứ tự state cố định:
  `[E_TR, E_LU, W_TR, W_LU, N_TR, N_LU, S_TR, S_LU] x 10 cells`.

Actions (4 green phases):
- A0 -> phase 0 (EW thẳng + rẽ phải)
- A1 -> phase 2 (NS thẳng + rẽ phải)
- A2 -> phase 4 (EW rẽ trái + quay đầu, KHÔNG thẳng)
- A3 -> phase 6 (NS rẽ trái + quay đầu, KHÔNG thẳng)

Phase semantics được kiểm tra tự động bằng controlledLinks + state string.
Nếu phase vi phạm (ví dụ phase 4/6 có thẳng), train/eval sẽ dừng và in lỗi.

Metrics:
- w_t: cumulative waiting time của tất cả xe trên incoming lanes.
- nwt: negative cumulative waiting time = tổng reward âm trong episode.
- vqs: cumulative vehicle queue size = tổng số xe dừng trên incoming lanes mỗi decision tick.

## 4) Scenario

Scenario hiện tại:
- `scenario/project_scenario`
- `scenario/project_scenario/osm.sumocfg`
- TLS ID nằm trong `experiment_config.yaml`.
- sumocfg chỉ chứa net; routes được inject từ scripts (per seed).

## 5) Lệnh chạy (PowerShell)

### 5.0 Kiểm tra nhanh (khuyến nghị)

```powershell
python .\tools\inspect_tls_and_lanes.py --sumocfg .\scenario\project_scenario\osm.sumocfg --tls-id GS_cluster_6164917307_6164917332_6164917334_6164917335 --gui 0
python .\tools\smoke_test_phase_control.py --sumocfg .\scenario\project_scenario\osm.sumocfg --tls-id GS_cluster_6164917307_6164917332_6164917334_6164917335 --seed 0 --log-every 5
```

Nếu phase semantics fail (có thẳng trong phase 4/6), hãy sửa TLS program trước khi train/eval.

### 5.1 Generate routes cho eval

```powershell
python .\tools\gen_routes.py --config .\experiment_config.yaml --seeds (0..99)
```

Routes lưu vào `results/routes` (bị ignore bởi git).

Lưu ý: nếu net không có u-turn connections, generator sẽ tự động fallback về straight/turn.
Nếu depart_speed="auto" thì route file không ghi `departSpeed` (SUMO tự chọn).

### 5.2 Train DQN (3 runs x 100 episodes)

```powershell
python .\pbl3_paper\train_dqn.py --config .\experiment_config.yaml
```

Nếu chỉ muốn chạy 1 run:

```powershell
python .\pbl3_paper\train_dqn.py --config .\experiment_config.yaml --run-start 2 --run-end 2
```

Outputs:
- `results/training/run1.csv`
- `results/training/run2.csv`
- `results/training/run3.csv`
- `results/training/run{1..3}_model.keras`

Training tự động tạo routes mỗi episode và lưu trong `results/routes/run1`, `results/routes/run2`, `results/routes/run3`.

Theo dõi tiến độ:
- `results/training/progress.txt` (episode hiện tại + seed)

### 5.2b Vẽ đồ thị training (avg_nwt/avg_vqs)

```powershell
python .\pbl3_paper\plot_training_avg.py --training-dir .\results\training --runs 1 2 3
```

Outputs:
- `results/training/avg_nwt.png`
- `results/training/avg_vqs.png`

### 5.3 Evaluate (FDS vs Adaptive) 100 simulations

```powershell
python .\pbl3_paper\eval.py --config .\experiment_config.yaml
```

Nếu muốn chọn model khác:

```powershell
python .\pbl3_paper\eval.py --config .\experiment_config.yaml --model .\results\training\run3_model.keras
```

Outputs:
- `results/eval/eval.csv` (100 dòng: fds vs adaptive per seed)
- `results/eval/eval_nwt.png`
- `results/eval/eval_vqs.png`
- `results/eval/eval_hist.png`
- `results/eval/stats.txt` (mean/std + paired t-test)

### 5.4 Chạy DQN trong SUMO-GUI (quan sát trực tiếp)

```powershell
python .\tools\run_dqn_gui.py --config .\experiment_config.yaml --model .\results\training\run3_model.keras
```

Mỗi lần chạy sẽ tạo route mới (seed theo thời gian) trong `results/routes/vis/`.

## 6) Lưu ý và lỗi thường gặp

- TLS ID sai: cập nhật `project.tls_id` trong `experiment_config.yaml`.
- Phase semantics fail: sửa TLS program hoặc phase mapping, rồi chạy lại.
- Không có incoming lanes: TLS id không điều khiển junction mong muốn.
- SUMO không tìm thấy: set `SUMO_HOME` hoặc thêm `SUMO/bin` vào `PATH`.

## 7) Cấu trúc file chính

```
PBL3/
  experiment_config.yaml
  README.md
  pbl3_paper/
    sumo_lane_cells.py
    env_sumo_cells.py
    baseline_fds.py
    train_dqn.py
    plot_training_avg.py
    eval.py
    requirements.txt
  tools/
    gen_routes.py
    inspect_tls_and_lanes.py
    smoke_test_phase_control.py
    run_dqn_gui.py
  scenario/
    project_scenario/
      osm.sumocfg
      osm.net.xml.gz
```
