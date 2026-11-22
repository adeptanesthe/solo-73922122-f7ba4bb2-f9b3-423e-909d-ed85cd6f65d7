"""API 端到端测试：上传开新批、登记/更正/撤回、批次令牌、并发两窗口、门控。"""
from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.csv_utils import MAX_MACHINES, parse_machines_csv
from app.main import app
from app.storage import store


@pytest.fixture(autouse=True)
async def _reset_store():
    async with store.lock:
        store.reset_locked([])
    yield
    async with store.lock:
        store.reset_locked([])


def _csv(text: str) -> bytes:
    return text.encode("utf-8")


def _upload(client: AsyncClient, content: bytes, name: str = "b.csv"):
    return client.post(
        "/api/batch", files={"file": (name, content, "text/csv")}
    )


CSV_9 = (
    "机器编号,初判\n"
    "M1,电源\nM2,电源\nM3,驱动\nM4,驱动\nM5,控制\nM6,控制\n"
    "M7,电源\nM8,驱动\nM9,控制\n"
)
CSV_6 = (
    "机器编号,初判\n"
    "M1,电源\nM2,驱动\nM3,控制\nM4,电源\nM5,驱动\nM6,控制\n"
)


def test_parse_csv_aliases_and_limits() -> None:
    machines = parse_machines_csv(
        _csv("机器编号,初判\nM1,电源\nM2,drive\nM3, 控制 \n")
    )
    assert [m["id"] for m in machines] == ["M1", "M2", "M3"]
    assert [m["initial"] for m in machines] == [
        "power",
        "drive",
        "control",
    ]
    assert all(m["inspected"] is None for m in machines)

    machines = parse_machines_csv(_csv("id,initial\nA,power\nB,control\n"))
    assert machines[0]["id"] == "A"

    for bad in [
        "编号,初判\n,电源\n",
        "机器编号,初判\nM1,电源\nM1,驱动\n",
        "机器编号,初判\nM1,爆炸\n",
        "机器编号\nM1\n",
        "\n",
    ]:
        with pytest.raises(Exception):
            parse_machines_csv(_csv(bad))

    over = "机器编号,初判\n" + "".join(
        f"M{i},电源\n" for i in range(MAX_MACHINES + 1)
    )
    with pytest.raises(Exception):
        parse_machines_csv(_csv(over))


async def test_full_workflow_inspect_correct_withdraw_and_gates() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        # 未上传时取批次 -> 404
        r = await client.get("/api/batch")
        assert r.status_code == 404

        # 9 台，初判多样，便于凑齐满秩样本
        r = await _upload(client, _csv(CSV_9))
        assert r.status_code == 200
        snap = r.json()
        assert snap["estimate"]["n_total"] == 9
        assert snap["estimate"]["n_uninspected"] == 9
        assert not snap["estimate"]["available"]  # 零实检 -> 撤下
        batch_id = snap["batch_id"]
        assert batch_id  # 快照必须带批次 ID

        # 写入不带批次令牌 -> 422（请求体校验失败）
        r = await client.post(
            "/api/machines/M1/inspect", json={"category": "power"}
        )
        assert r.status_code == 422

        # 非法类别 400
        r = await client.post(
            "/api/machines/M1/inspect",
            json={"category": "nuclear", "batch_id": batch_id},
        )
        assert r.status_code == 400
        # 不存在的机器 400
        r = await client.post(
            "/api/machines/NOPE/inspect",
            json={"category": "power", "batch_id": batch_id},
        )
        assert r.status_code == 400

        # 登记：让每类实检样本包含不同初判，保证撤回后仍可凑满秩
        plan = {
            "M1": "power", "M2": "power", "M3": "power",
            "M4": "drive", "M5": "drive", "M6": "drive",
            "M7": "control", "M8": "control", "M9": "control",
        }
        for mid, cat in plan.items():
            r = await client.post(
                f"/api/machines/{mid}/inspect",
                json={"category": cat, "batch_id": batch_id},
            )
            assert r.status_code == 200
            # 成功响应携带同一批次 ID
            assert r.json()["batch_id"] == batch_id

        snap = (await client.get("/api/batch")).json()
        est = snap["estimate"]
        assert est["n_inspected"] == 9 and est["n_uninspected"] == 0
        assert est["available"] and est["actual_total"] == {
            "power": 3, "drive": 3, "control": 3
        }

        # 更正 M1：power -> control，实检数随之变化
        r = await client.post(
            "/api/machines/M1/inspect",
            json={"category": "control", "batch_id": batch_id},
        )
        est = r.json()["estimate"]
        assert est["actual_total"] == {
            "power": 2, "drive": 3, "control": 4
        }

        # 撤回 M1：重新算作未检；此时有未检件且三类实检样本仍齐、矩阵满秩
        r = await client.post(
            "/api/machines/M1/withdraw", json={"batch_id": batch_id}
        )
        est = r.json()["estimate"]
        assert est["n_inspected"] == 8 and est["n_uninspected"] == 1
        assert est["available"]
        m1 = next(m for m in r.json()["machines"] if m["id"] == "M1")
        assert m1["inspected"] is None
        # 估数 = 实检数 + 未检折算，总和守恒
        assert sum(est["estimated_total"].values()) == pytest.approx(9.0)
        for c in ("power", "drive", "control"):
            assert est["estimated_total"][c] == pytest.approx(
                est["inspected_counts"][c]
                + est["estimated_uninspected"][c]
            )

        # 继续撤回至某类实检样本归零 -> 估数撤下并提示补检
        for mid in ("M7", "M8", "M9"):
            r = await client.post(
                f"/api/machines/{mid}/withdraw", json={"batch_id": batch_id}
            )
        est = r.json()["estimate"]
        assert not est["available"]
        assert "控制" in est["reason"]


async def test_each_upload_creates_a_new_batch_even_for_same_csv() -> None:
    """每次成功上传都另开一批；重传同一 CSV 也分配新批次并清空记录。"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        r1 = await _upload(client, _csv(CSV_6))
        first = r1.json()
        bid1 = first["batch_id"]
        assert first["estimate"]["n_inspected"] == 0

        # 旧批里登记一台
        r = await client.post(
            "/api/machines/M1/inspect",
            json={"category": "power", "batch_id": bid1},
        )
        assert r.status_code == 200
        assert r.json()["estimate"]["n_inspected"] == 1

        # 重传同一份 CSV：必须开新批，拆检记录全部归零
        r2 = await _upload(client, _csv(CSV_6))
        second = r2.json()
        bid2 = second["batch_id"]
        assert bid2 != bid1
        assert second["estimate"]["n_inspected"] == 0
        assert all(m["inspected"] is None for m in second["machines"])

        # 再传一次（仍是同一文件）：批次号继续递增
        r3 = await _upload(client, _csv(CSV_6))
        assert r3.json()["batch_id"] != bid2
        assert r3.json()["estimate"]["n_inspected"] == 0

        # GET 始终只返回当前批次
        cur = (await client.get("/api/batch")).json()
        assert cur["batch_id"] == r3.json()["batch_id"]


async def test_stale_batch_write_is_rejected_and_never_hits_new_batch() -> None:
    """旧页面拿旧批次号登记同号机器：409 拒绝，新批同号机器保持未检。"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        # 批次 A：登记 M1 -> power
        r = await _upload(client, _csv(CSV_6))
        a = r.json()
        bid_a = a["batch_id"]
        r = await client.post(
            "/api/machines/M1/inspect",
            json={"category": "power", "batch_id": bid_a},
        )
        assert r.status_code == 200

        # 另一窗口换批：上传批次 B（机器编号同样含 M1）
        r = await _upload(client, _csv(CSV_6))
        b = r.json()
        bid_b = b["batch_id"]
        assert bid_b != bid_a
        assert all(m["inspected"] is None for m in b["machines"])

        # 旧窗口迟到的登记请求（带 A 的批次号、写同号机器 M1）
        r = await client.post(
            "/api/machines/M1/inspect",
            json={"category": "control", "batch_id": bid_a},
        )
        assert r.status_code == 409
        body = r.json()
        assert body["code"] == "batch_switched"
        assert body["current_batch_id"] == bid_b
        assert "批次已切换" in body["detail"]

        # 新批 M1 必须仍未检，且没有任何旧批记录泄漏
        cur = (await client.get("/api/batch")).json()
        assert cur["batch_id"] == bid_b
        m1 = next(m for m in cur["machines"] if m["id"] == "M1")
        assert m1["inspected"] is None
        assert cur["estimate"]["n_inspected"] == 0
        # 缺实检样本时估数照旧撤下
        assert cur["estimate"]["available"] is False

        # 旧批次号撤回同样被 409 拒绝
        r = await client.post(
            "/api/machines/M2/withdraw", json={"batch_id": bid_a}
        )
        assert r.status_code == 409
        assert r.json()["current_batch_id"] == bid_b

        # 旧批次号 + 新批里不存在的编号：仍优先报换批 409，而不是 400
        r = await client.post(
            "/api/machines/GHOST/inspect",
            json={"category": "power", "batch_id": bid_a},
        )
        assert r.status_code == 409

        # 用当前批次号登记 -> 正常生效，估数只按新批记录计算
        r = await client.post(
            "/api/machines/M1/inspect",
            json={"category": "control", "batch_id": bid_b},
        )
        assert r.status_code == 200
        m1 = next(m for m in r.json()["machines"] if m["id"] == "M1")
        assert m1["inspected"] == "control"
        assert r.json()["batch_id"] == bid_b


async def test_estimates_are_computed_per_current_batch_only() -> None:
    """混淆矩阵/q/估数只来自当前批次拆检记录，旧批样本不得带入。"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        # 批次 A：做一批满秩拆检，估数有效
        r = await _upload(client, _csv(CSV_9))
        bid_a = r.json()["batch_id"]
        plan = {
            "M1": "power", "M2": "power", "M3": "power",
            "M4": "drive", "M5": "drive", "M6": "drive",
            "M7": "control", "M8": "control", "M9": "control",
        }
        for mid, cat in plan.items():
            await client.post(
                f"/api/machines/{mid}/inspect",
                json={"category": cat, "batch_id": bid_a},
            )
        # 撤回一台制造未检件，此时三类样本齐、矩阵满秩 -> 估数有效
        r = await client.post(
            "/api/machines/M1/withdraw", json={"batch_id": bid_a}
        )
        assert r.json()["estimate"]["available"] is True

        # 换批 B：新批零拆检，估数必须立即撤下（旧批样本不能延续）
        r = await _upload(client, _csv(CSV_6))
        bid_b = r.json()["batch_id"]
        est = r.json()["estimate"]
        assert est["n_inspected"] == 0
        assert est["available"] is False
        assert "缺少实检类别样本" in est["reason"]
        assert est["confusion_matrix"] is None

        # 在新批只登记一类实检：仍缺两类样本，继续撤下
        await client.post(
            "/api/machines/M1/inspect",
            json={"category": "power", "batch_id": bid_b},
        )
        est = (await client.get("/api/batch")).json()["estimate"]
        assert est["available"] is False
        assert "驱动" in est["reason"] and "控制" in est["reason"]
        # 已有的部分计数只反映新批这一台
        assert est["inspected_counts"] == {
            "power": 1, "drive": 0, "control": 0
        }


async def test_concurrent_two_windows_both_take_effect() -> None:
    """模拟两个窗口同时改不同机器（以及同一台），两笔都须生效。"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        r = await _upload(client, _csv(CSV_6))
        batch_id = r.json()["batch_id"]

        # 不同机器并发登记：两笔都生效
        await asyncio.gather(
            client.post("/api/machines/M1/inspect",
                        json={"category": "power", "batch_id": batch_id}),
            client.post("/api/machines/M2/inspect",
                        json={"category": "drive", "batch_id": batch_id}),
            client.post("/api/machines/M3/inspect",
                        json={"category": "control", "batch_id": batch_id}),
        )
        snap = (await client.get("/api/batch")).json()
        by_id = {m["id"]: m["inspected"] for m in snap["machines"]}
        assert by_id["M1"] == "power"
        assert by_id["M2"] == "drive"
        assert by_id["M3"] == "control"

        # 同一台机器两个窗口同时提交不同类别：两次写入都不丢，
        # 最终状态是其中之一（按锁顺序），且快照自洽
        r1, r2 = await asyncio.gather(
            client.post("/api/machines/M1/inspect",
                        json={"category": "drive", "batch_id": batch_id}),
            client.post("/api/machines/M1/inspect",
                        json={"category": "control", "batch_id": batch_id}),
        )
        assert r1.status_code == r2.status_code == 200
        final = (await client.get("/api/batch")).json()
        m1 = next(m for m in final["machines"] if m["id"] == "M1")
        assert m1["inspected"] in {"drive", "control"}
        # 实检数与机器明细始终一致（同一份拆检记录）
        counts = final["estimate"]["inspected_counts"]
        actual = {"power": 0, "drive": 0, "control": 0}
        for m in final["machines"]:
            if m["inspected"]:
                actual[m["inspected"]] += 1
        assert counts == actual

        # 并发撤回 + 登记不同机器，同样都生效
        await asyncio.gather(
            client.post("/api/machines/M2/withdraw",
                        json={"batch_id": batch_id}),
            client.post("/api/machines/M4/inspect",
                        json={"category": "power", "batch_id": batch_id}),
        )
        snap = (await client.get("/api/batch")).json()
        by_id = {m["id"]: m["inspected"] for m in snap["machines"]}
        assert by_id["M2"] is None and by_id["M4"] == "power"

        # 换批瞬间两窗口并发：旧批令牌的写请求全部 409，新批不被污染
        r = await _upload(client, _csv(CSV_6))
        new_batch_id = r.json()["batch_id"]
        results = await asyncio.gather(
            client.post("/api/machines/M1/inspect",
                        json={"category": "power", "batch_id": batch_id}),
            client.post("/api/machines/M2/withdraw",
                        json={"batch_id": batch_id}),
            client.post("/api/machines/M3/inspect",
                        json={"category": "control",
                              "batch_id": new_batch_id}),
        )
        assert results[0].status_code == 409
        assert results[1].status_code == 409
        assert results[2].status_code == 200
        snap = (await client.get("/api/batch")).json()
        by_id = {m["id"]: m["inspected"] for m in snap["machines"]}
        assert by_id["M1"] is None and by_id["M2"] is None
        assert by_id["M3"] == "control"
