"""DLC 整理端点。

POST /api/dlc
  可传 games 列表(推荐,先经 /api/library 解析),或传 mode+value 直接解析。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas import Game
from app.steam.dlc import build_dlc_report
from app.steam.importer import resolve_library

router = APIRouter()


class DlcRequest(BaseModel):
    games: list[Game] | None = None
    mode: str | None = None
    value: str | None = None


@router.post("/dlc")
async def get_dlc_report(req: DlcRequest) -> dict:
    try:
        if req.games:
            games = [g.model_dump() for g in req.games]
        elif req.mode and req.value:
            games = await resolve_library(req.mode, req.value)
        else:
            raise HTTPException(status_code=400, detail="请提供 games 列表或 mode+value")
        return await build_dlc_report(games)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"生成 DLC 报告失败: {exc}") from exc
