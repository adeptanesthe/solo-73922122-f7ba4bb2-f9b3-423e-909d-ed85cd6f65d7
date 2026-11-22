"""FastAPI 入口：CSV 上传、拆检登记/更正/撤回、估算快照。

换批规则
========
* 每次成功上传 CSV 都建立一个新批次（新批次 ID、清空拆检记录），
  重传同一份文件也不例外。
* 登记/更正/撤回请求必须携带点击时页面显示的 ``batch_id``；若批次
  已被其他窗口的上传替换，返回 HTTP 409 与当前批次 ID，拒绝写入。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .csv_utils import CsvError, parse_machines_csv
from .storage import BatchSwitchedError, StoreError, store

app = FastAPI(title="返修机故障占比估算", version="1.1.0")

STATIC_DIR = Path(__file__).parent / "static"


class InspectPayload(BaseModel):
    category: str = Field(..., description="实检类别：power/drive/control")
    batch_id: str = Field(..., description="操作时页面显示的批次 ID")


class WithdrawPayload(BaseModel):
    batch_id: str = Field(..., description="操作时页面显示的批次 ID")


@app.exception_handler(CsvError)
@app.exception_handler(StoreError)
async def _business_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(BatchSwitchedError)
async def _batch_switched_handler(
    request: Request, exc: BatchSwitchedError
) -> JSONResponse:
    # 409：写入针对的批次已过期。带回当前批次 ID，供页面提示并重新同步。
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
            "code": "batch_switched",
            "current_batch_id": exc.current_batch_id,
        },
    )


@app.post("/api/batch")
async def upload_batch(request: Request) -> dict:
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "filename"):
        raise CsvError("请通过 file 字段上传 CSV 文件")
    content = await upload.read()
    if not content:
        raise CsvError("上传的文件为空")
    machines = parse_machines_csv(content)
    async with store.lock:
        # 每次成功上传都另开一批：新批次 ID + 清空旧拆检记录。
        store.reset_locked(machines)
        return store.snapshot_locked()


@app.get("/api/batch")
async def get_batch() -> dict:
    async with store.lock:
        if store.is_empty_locked():
            return JSONResponse(
                status_code=404,
                content={"detail": "尚未上传批次，请先上传 CSV"},
            )
        return store.snapshot_locked()


@app.post("/api/machines/{machine_id}/inspect")
async def inspect_machine(machine_id: str, payload: InspectPayload) -> dict:
    # 批次/机器/类别的校验顺序统一由存储层把关：批次过期优先返回 409，
    # 避免旧批请求因非法类别先收到 400 而看不到换批提示。
    async with store.lock:
        return store.update_inspection_locked(
            machine_id, payload.category, payload.batch_id
        )


@app.post("/api/machines/{machine_id}/withdraw")
async def withdraw_machine(machine_id: str, payload: WithdrawPayload) -> dict:
    async with store.lock:
        return store.update_inspection_locked(
            machine_id, None, payload.batch_id
        )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)
