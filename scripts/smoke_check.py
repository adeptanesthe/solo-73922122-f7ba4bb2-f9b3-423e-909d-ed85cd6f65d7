"""对运行中的网页服务做端到端冒烟（纯标准库，可在容器外执行）。

覆盖：CSV 上传（每次开新批）、登记/更正/撤回、估数门控（缺类/不满秩
撤下、补检恢复）、全检实数、台数守恒、两个“窗口”并发修改不同机器
两笔都生效，以及换批隔离：旧批令牌写入被 409 拒绝且新批同号机器不被
污染、迟到旧批响应不影响当前快照。

用法：python scripts/smoke_check.py [base_url] [csv_path]
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import urllib.request
import urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
CSV_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/sample_machines.csv"
CATS = ("power", "drive", "control")


def call(method: str, path: str, body=None, raw: bytes | None = None,
         ctype: str | None = None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        data = raw
        headers["Content-Type"] = ctype or "application/octet-stream"
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise SystemExit(1)


def upload(csv_path: str):
    with open(csv_path, "rb") as f:
        boundary = "----smokeboundary"
        payload = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"sample.csv\"\r\nContent-Type: text/csv\r\n\r\n"
        ).encode() + f.read() + f"\r\n--{boundary}--\r\n".encode()
    return call("POST", "/api/batch", raw=payload,
                ctype=f"multipart/form-data; boundary={boundary}")


def inspect(mid: str, cat: str, batch_id: str):
    return call("POST", f"/api/machines/{mid}/inspect",
                {"category": cat, "batch_id": batch_id})


def withdraw(mid: str, batch_id: str):
    return call("POST", f"/api/machines/{mid}/withdraw",
                {"batch_id": batch_id})


# 1. 上传
status, snap = upload(CSV_PATH)
check("上传 CSV", status == 200, f"status={status}")
est = snap["estimate"]
n_total = est["n_total"]
check("初始全部未检且估数撤下（缺实检样本）",
      est["n_uninspected"] == n_total and not est["available"])
check("快照返回批次 ID", bool(snap["batch_id"]))

# 2. 构造满秩拆检：三个实检类别的初判分布互不相同（行向量线性无关）
#    power: (3,2,1)  drive: (1,3,2)  control: (2,1,3)
batch_id = snap["batch_id"]
machines = snap["machines"]
by_initial = {c: [m["id"] for m in machines if m["initial"] == c] for c in CATS}
layout = {
    "power": {"power": 3, "drive": 2, "control": 1},
    "drive": {"power": 1, "drive": 3, "control": 2},
    "control": {"power": 2, "drive": 1, "control": 3},
}
plan: dict[str, str] = {}
for true_cat, cells in layout.items():
    for init_cat, k in cells.items():
        for _ in range(k):
            mid = by_initial[init_cat].pop()
            plan[mid] = true_cat

for mid, cat in plan.items():
    status, s = inspect(mid, cat, batch_id)
    check(f"登记 {mid} -> {cat}", status == 200)
check("成功响应批次 ID 一致", s["batch_id"] == batch_id)
est = s["estimate"]
check("补检后估数恢复（满秩、三类齐）", est["available"],
      f"reason={est.get('reason')}")
check("台数守恒（估数总和=总台数）",
      abs(sum(est["estimated_total"].values()) - n_total) < 1e-8)
check("折算台数+实检数=全批估计", all(
    abs(est["estimated_total"][c] -
        (est["inspected_counts"][c] + est["estimated_uninspected"][c])) < 1e-9
    for c in CATS))
pi = est["pi"]
check("π 在单纯形内", all(x >= 0 for x in pi) and abs(sum(pi) - 1) < 1e-9)
print(f"       拟合 π = {dict(zip(CATS, (round(x, 4) for x in pi)))}，"
      f"SSE = {est['sse']:.6f}")

# 3. 更正一台：取一台已检机，改成另一类别，实检数应同步变化
mid = next(iter(plan))
old_counts = dict(est["inspected_counts"])
new_cat = "drive" if plan[mid] != "drive" else "power"
status, s = inspect(mid, new_cat, batch_id)
check("更正返回 200", status == 200)
row = next(m for m in s["machines"] if m["id"] == mid)
check("机器明细已更正", row["inspected"] == new_cat)
check("实检数随同一份记录更新",
      s["estimate"]["inspected_counts"][new_cat] == old_counts[new_cat] + 1)
plan[mid] = new_cat

# 4. 撤回一台：重新算作未检
status, s = withdraw(mid, batch_id)
row = next(m for m in s["machines"] if m["id"] == mid)
check("撤回后该台重算未检", status == 200 and row["inspected"] is None)
check("撤回后未检数 +1", s["estimate"]["n_uninspected"] == est["n_uninspected"] + 1)

# 5. 门控：撤回某实检类别的全部样本 -> 估数撤下并提示补检
status, snap_all = call("GET", "/api/batch")
target = "control"
target_label = {"power": "电源", "drive": "驱动", "control": "控制"}[target]
victims = [m["id"] for m in snap_all["machines"]
           if m["inspected"] == target and m["id"] != mid]
for v in victims:
    status, s = withdraw(v, batch_id)
check("缺类门控触发", not s["estimate"]["available"]
      and target_label in s["estimate"]["reason"],
      s["estimate"].get("reason"))

# 6. 补检一台恢复该类样本（若矩阵仍满秩，估数恢复）
status, s = inspect(victims[0], target, batch_id)
check("补检后接口正常", status == 200)
print(f"       补检后估数状态: available={s['estimate']['available']}"
      f" reason={s['estimate'].get('reason')}")

# 7. 并发：两个窗口同时改不同机器，两笔都生效
m_a, m_b = [m["id"] for m in s["machines"] if m["inspected"] is None][:2]
with cf.ThreadPoolExecutor(max_workers=2) as ex:
    f1 = ex.submit(inspect, m_a, "power", batch_id)
    f2 = ex.submit(inspect, m_b, "control", batch_id)
    r1, r2 = f1.result(), f2.result()
check("并发两请求均 200", r1[0] == 200 and r2[0] == 200)
status, final = call("GET", "/api/batch")
by_id = {m["id"]: m["inspected"] for m in final["machines"]}
check("两窗口修改不同机器均生效",
      by_id[m_a] == "power" and by_id[m_b] == "control")
counts = final["estimate"]["inspected_counts"]
actual = {c: 0 for c in CATS}
for m in final["machines"]:
    if m["inspected"]:
        actual[m["inspected"]] += 1
check("实检数与明细一致（同一份拆检记录）", counts == actual)

# 8. 换批流程：重传同一 CSV 必须另开一批，旧批令牌写入一律 409
old_batch_id = batch_id
status, new_snap = upload(CSV_PATH)
check("重传同一 CSV 成功开新批", status == 200)
new_batch_id = new_snap["batch_id"]
check("新批次 ID 与旧批不同", new_batch_id != old_batch_id)
check("新批拆检记录全部归零",
      all(m["inspected"] is None for m in new_snap["machines"])
      and new_snap["estimate"]["n_inspected"] == 0)
check("新批缺实检样本，估数照旧撤下",
      new_snap["estimate"]["available"] is False)

# 8a. 旧页面迟到的登记（同号机器）：409 拒绝，新批同号机器保持未检
stale_mid = next(m["id"] for m in new_snap["machines"])
status, err = inspect(stale_mid, "control", old_batch_id)
check("旧批登记被 409 拒绝", status == 409)
check("409 带回当前批次 ID 与换批提示",
      err.get("code") == "batch_switched"
      and err.get("current_batch_id") == new_batch_id
      and "批次已切换" in err.get("detail", ""))
status, cur = call("GET", "/api/batch")
row = next(m for m in cur["machines"] if m["id"] == stale_mid)
check("新批同号机器未被旧批写入污染", row["inspected"] is None)

# 8b. 旧批撤回同样 409
status, err = withdraw(stale_mid, old_batch_id)
check("旧批撤回被 409 拒绝", status == 409
      and err.get("current_batch_id") == new_batch_id)

# 8c. 换批瞬间并发：旧批令牌两笔 409，当前批令牌一笔 200 且生效
other_ids = [m["id"] for m in cur["machines"] if m["id"] != stale_mid][:2]
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    f1 = ex.submit(inspect, other_ids[0], "power", old_batch_id)
    f2 = ex.submit(withdraw, other_ids[1], old_batch_id)
    f3 = ex.submit(inspect, stale_mid, "drive", new_batch_id)
    r1, r2, r3 = f1.result(), f2.result(), f3.result()
check("换批并发：旧批两笔 409、当前批一笔 200",
      r1[0] == 409 and r2[0] == 409 and r3[0] == 200)
status, cur = call("GET", "/api/batch")
by_id = {m["id"]: m["inspected"] for m in cur["machines"]}
check("只有当前批那笔生效",
      by_id[stale_mid] == "drive"
      and by_id[other_ids[0]] is None
      and by_id[other_ids[1]] is None)
check("当前批估数仅按本批记录计算（仅 1 台实检 -> 缺类撤下）",
      cur["estimate"]["inspected_counts"] == {
          "power": 0, "drive": 1, "control": 0}
      and cur["estimate"]["available"] is False)

# 9. 全部拆完 -> 实数（在当前批次内重新做满）
status, snap_all = call("GET", "/api/batch")
bid = snap_all["batch_id"]
for m in snap_all["machines"]:
    if m["inspected"] is None:
        call("POST", f"/api/machines/{m['id']}/inspect",
             {"category": m["initial"], "batch_id": bid})
status, done = call("GET", "/api/batch")
e = done["estimate"]
check("全部拆完直接给实数",
      e["n_uninspected"] == 0 and e["available"] and e["pi"] is None)
check("实数合计=总台数", sum(e["actual_total"].values()) == n_total)
print(f"       实检实数: {e['actual_total']}")
print("\n全部冒烟检查通过 ✔")
