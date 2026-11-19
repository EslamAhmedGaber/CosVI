"""Write the GPU and CPU job queues of the full campaign requested by the author on 13 Nov ("all tests at all values").
Run once:  python make_queues.py   then start   python queue_runner.py queue_gpu.jsonl status_gpu.jsonl   (and cpu)."""
import json, os

WORK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PY = r"python"
PYJ = os.path.join(WORK, "envs", "jax_mafbm", "Scripts", "python.exe")
EXP = os.path.join(WORK, "code", "experiments")
COMP = os.path.join(WORK, "code", "competitor")
R = os.path.join(WORK, "results")
PORT_OK = os.path.join(R, "g0", "PORT_OK.flag")


def job(name, cwd, cmd, log, **kw):
    d = {"name": name, "cwd": cwd, "cmd": cmd, "log": log}; d.update(kw); return d


gpu = [job("wait_for_n4", EXP, [PY, "-c", "pass"], os.path.join(R, "queue_gpu.log"),
           wait_file=os.path.join(R, "n4", "n4_markov_tuned_seed2_s0.1.json")),
       job("n3_bridge_ours_all", EXP, [PY, "-u", "n3_bridge_ours.py"], os.path.join(R, "n3", "n3_run.log")),
       # moved forward on 13 Nov after N1b seed 0: is the exact grid's deficit only optimisation speed?
       job("n1b_budget_ablation_H075", EXP, [PY, "-u", "n1b_representations.py", "--reps", "grid", "twoscale", "cosine", "markov_tuned",
                                             "--H", "0.75", "--seeds", "0", "--steps", "10000", "--tag", "_steps10000"],
           os.path.join(R, "n1b", "n1b_ablation_run.log")),
       job("n1c_competitor_system_all", EXP, [PY, "-u", "n1c_competitor_system.py", "--seeds", "0", "1", "2"],
           os.path.join(R, "n1c", "n1c_run.log"), if_flag=PORT_OK),
       # added 14 Nov after N4 v1 (resolution rule violated by J = 32; grid-coordinate yardstick failing at high H)
       job("n4v2_all", EXP, [PY, "-u", "n4v2_nile.py", "--seeds", "0", "1", "2"], os.path.join(R, "n4", "n4v2_run.log")),
       # 14 Nov: first N1c job failed (host sync inside CUDA-graph capture); fixed with an on-device omega(H) bank
       job("n1c_competitor_system_all_v2", EXP, [PY, "-u", "n1c_competitor_system.py", "--seeds", "0", "1", "2"],
           os.path.join(R, "n1c", "n1c_run_v2.log"), if_flag=PORT_OK),
       # 14 Nov fairness: the competitor with its BEST configuration (12 nodes, long history, stable gamma_max,
       # standard-fBm scale), as their full system (N1c, bridge) and as a representation under our inference (N1b)
       job("n1c_competitor_best_all", EXP, [PY, "-u", "n1c_competitor_system.py", "--seeds", "0", "1", "2", "--config", "best"],
           os.path.join(R, "n1c", "n1c_run_best.log"), if_flag=PORT_OK),
       # 14 Nov: their joint H learning did not move from its initial value; ablation with a separate, larger lr for H
       job("n1c_joint_lrH_ablation", EXP, [PY, "-u", "n1c_competitor_system.py", "--modes", "joint", "--seeds", "0", "1", "2", "--lr-h", "0.02"],
           os.path.join(R, "n1c", "n1c_run_lrH.log"), if_flag=PORT_OK),
       # 14 Nov 18:50 reorder (scheduling only, no criterion changed): our side's seed spread (N1b seeds 1-2) is needed
       # before any N1b/N1c claim, so it runs before the fairness extras; the port bridge grid (N3 already decided
       # against their own code at all four H) goes last.
       job("n1b_seed1", EXP, [PY, "-u", "n1b_representations.py", "--seeds", "1", "--steps", "5000"], os.path.join(R, "n1b", "n1b_seed1_run.log")),
       job("n1b_seed2", EXP, [PY, "-u", "n1b_representations.py", "--seeds", "2", "--steps", "5000"], os.path.join(R, "n1b", "n1b_seed2_run.log")),
       job("n1b_markov_best_seed0", EXP, [PY, "-u", "n1b_representations.py", "--reps", "markov_best", "--seeds", "0", "--steps", "5000"],
           os.path.join(R, "n1b", "n1b_markov_best_run.log"))]
for H in (0.6, 0.7, 0.8, 0.9):
    gpu.append(job(f"port_bridge_best_H{H}_s0", COMP, [PY, "-u", "g0_port_validation.py", "--hurst", str(H), "--seed", "0", "--skip-prior",
                                                       "--K", "12", "--gmax", "40", "--gmin", "1e-6", "--std-scale", "--tag", "_best"],
                   os.path.join(R, "g0", f"port_validation_H{H:.2f}_seed0_best.log"), if_flag=PORT_OK))
gpu.append(job("n4c_competitor_nile_all", EXP, [PY, "-u", "n4c_competitor_nile.py", "--seeds", "0", "1", "2"],
               os.path.join(R, "n4", "n4c_run.log"), if_flag=PORT_OK))
gpu += [job("n1b_budget_ablation_rest", EXP, [PY, "-u", "n1b_representations.py", "--reps", "grid", "twoscale", "cosine", "markov_tuned",
                                              "--H", "0.6", "0.9", "--seeds", "0", "--steps", "10000", "--tag", "_steps10000"],
            os.path.join(R, "n1b", "n1b_ablation_run.log")),
        job("n4v2_sensitivity_s005", EXP, [PY, "-u", "n4v2_nile.py", "--reps", "grid", "twoscale", "cosine64", "--seeds", "0", "--s", "0.05"],
            os.path.join(R, "n4", "n4v2_sens_run.log")),
        job("n4v2_sensitivity_s02", EXP, [PY, "-u", "n4v2_nile.py", "--reps", "grid", "twoscale", "cosine64", "--seeds", "0", "--s", "0.2"],
            os.path.join(R, "n4", "n4v2_sens_run.log"))]
for H in (0.6, 0.7, 0.8, 0.9):
    for s in (0, 1, 2):
        if H == 0.7 and s == 0:
            continue
        gpu.append(job(f"port_bridge_H{H}_s{s}", COMP, [PY, "-u", "g0_port_validation.py", "--hurst", str(H), "--seed", str(s), "--skip-prior"],
                       os.path.join(R, "g0", f"port_validation_H{H:.2f}_seed{s}.log"), if_flag=PORT_OK))

cpu = [job("wait_for_g0_H07", COMP, [PY, "-c", "pass"], os.path.join(R, "queue_cpu.log"),
           wait_file=os.path.join(R, "g0", "bridge_H0.70_theta1.0_seed0_steps2000.json"))]
# 14 Nov 22:30: theta = 0 at the other three H added (the CPU was idle; completes N3's theta = 0 column against their own code)
for H, th in ((0.9, 1), (0.6, 1), (0.8, 1), (0.7, 0), (0.9, 0), (0.6, 0), (0.8, 0)):
    cpu.append(job(f"their_code_bridge_H{H}_theta{th}", COMP,
                   [PYJ, "-u", "g0_mafbm_bridge.py", "--hurst", str(H), "--theta", str(th), "--seed", "0", "--steps", "2000"],
                   os.path.join(R, "g0", f"bridge_H{H:.2f}_theta{th}_seed0.log")))

for fn, q in (("queue_gpu.jsonl", gpu), ("queue_cpu.jsonl", cpu)):
    with open(os.path.join(WORK, "code", fn), "w", encoding="utf-8") as fh:
        for j in q:
            fh.write(json.dumps(j) + "\n")
    print(fn, len(q), "jobs")
