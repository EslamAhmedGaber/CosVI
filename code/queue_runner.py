"""Sequential job runner for long experiment campaigns (one runner per device).

Usage:  python queue_runner.py <queue.jsonl> <status.jsonl>
Each line of the queue is a JSON object:
  {"name": "...", "cwd": "...", "cmd": ["exe", "arg", ...], "log": "path",
   "wait_file": "optional path to wait for before running",
   "if_flag": "optional path; the job is skipped (and recorded as skipped) unless this file exists"}
The queue file is re-read before every job, so jobs can be appended while the runner works. A job is identified by
its "name"; finished or skipped names are recorded in the status file and never rerun.
"""
import json, os, subprocess, sys, time


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


def main():
    qfile, sfile = sys.argv[1], sys.argv[2]
    while True:
        done = {s["name"] for s in load(sfile)}
        pending = [j for j in load(qfile) if j["name"] not in done]
        if not pending:
            print("queue empty", flush=True)
            return
        job = pending[0]
        if job.get("wait_file"):
            while not os.path.exists(job["wait_file"]):
                time.sleep(60)
        rec = {"name": job["name"], "start": time.strftime("%Y-%m-%d %H:%M:%S")}
        if job.get("if_flag") and not os.path.exists(job["if_flag"]):
            rec.update({"status": "skipped (flag missing)", "end": rec["start"]})
        else:
            os.makedirs(os.path.dirname(job["log"]), exist_ok=True)
            with open(job["log"], "a", encoding="utf-8") as lg:
                p = subprocess.run(job["cmd"], cwd=job["cwd"], stdout=lg, stderr=subprocess.STDOUT)
            rec.update({"status": f"exit {p.returncode}", "end": time.strftime("%Y-%m-%d %H:%M:%S")})
        with open(sfile, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
