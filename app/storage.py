"""批次与拆检记录的内存存储。

演示环境使用单进程内存存储，所有读写经过同一把异步锁，保证两个浏览器
窗口同时登记/更正/撤回（即便操作同一台机器）时两笔修改都基于同一份
拆检记录顺序生效，实检数与估数始终来自同一快照。

批次隔离
========
* 每次成功上传 CSV 都会建立一个**新批次**，分配单调递增的批次 ID
  （``B0001``、``B0002`` ……），并重整机台清单与全部拆检记录；
  重传同一份 CSV 也一样开新批。
* 登记/更正/撤回必须携带操作时页面显示的批次 ID。若写入前批次已被
  替换（另一窗口上传了新 CSV），抛出 :class:`BatchSwitchedError`，
  路由层映射为 HTTP 409 并带回当前批次 ID，旧批的写入不会落到新批
  同号机器上。
* 混淆矩阵、未检件初判分布与估计台数始终只按**当前批次**的拆检记录
  计算（快照在持锁期间由同一份机台清单生成）。

注意：该实现与 uvicorn 单 worker 绑定（docker-compose 已固定
``--workers 1``）；多进程部署需替换为数据库等共享存储。
"""
from __future__ import annotations

import asyncio

from .estimation import CATEGORIES, CAT_INDEX, estimate


class StoreError(Exception):
    """业务校验错误，映射为 HTTP 400。"""


class BatchSwitchedError(StoreError):
    """写入针对的批次已不是当前批次，映射为 HTTP 409。"""

    def __init__(self, current_batch_id: str) -> None:
        self.current_batch_id = current_batch_id
        super().__init__(
            f"批次已切换，当前批次为 {current_batch_id}，"
            "本次登记/更正/撤回未生效，请按当前批次重新操作。"
        )


class BatchStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._machines: list[dict] = []
        self._id_index: dict[str, int] = {}
        self._batch_seq = 0
        self._batch_id: str | None = None

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def reset_locked(self, machines: list[dict]) -> None:
        """载入新批次（调用方须持锁）。

        每次调用都分配新的批次 ID 并丢弃全部旧拆检记录；``machines``
        为空时回到“尚未上传”状态（测试清场用）。
        """
        self._machines = machines
        self._id_index = {m["id"]: i for i, m in enumerate(machines)}
        if machines:
            self._batch_seq += 1
            self._batch_id = f"B{self._batch_seq:04d}"
        else:
            self._batch_id = None

    def current_batch_id_locked(self) -> str | None:
        return self._batch_id

    def snapshot_locked(self) -> dict:
        """基于同一份拆检记录生成批次 ID、机器清单与估算结果（持锁调用）。"""
        result = estimate(self._machines)
        return {
            "batch_id": self._batch_id,
            "machines": [dict(m) for m in self._machines],
            "estimate": result.to_payload(),
        }

    def update_inspection_locked(
        self, machine_id: str, category: str | None, batch_id: str
    ) -> dict:
        """登记/更正（category 为三类之一）或撤回（None）一台机器。

        ``batch_id`` 必须等于当前批次 ID，否则抛
        :class:`BatchSwitchedError`，拒绝任何写入。
        """
        # 先校验批次：旧批请求即使带着同号机器编号也不能落到新批。
        if self._batch_id is None:
            raise StoreError("尚未上传批次，请先上传 CSV")
        if batch_id != self._batch_id:
            raise BatchSwitchedError(self._batch_id)
        if machine_id not in self._id_index:
            raise StoreError(f"机器编号 {machine_id} 不存在于当前批次")
        if category is not None and category not in CAT_INDEX:
            raise StoreError(
                f"非法实检类别 {category!r}，允许：{', '.join(CATEGORIES)}"
            )
        machine = self._machines[self._id_index[machine_id]]
        machine["inspected"] = category
        return self.snapshot_locked()

    def is_empty_locked(self) -> bool:
        return not self._machines


store = BatchStore()
