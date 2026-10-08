"""游戏库导入端点。

POST /api/library  body: {"mode": "steamid", "value": "<steamid64>"}
需要 STEAM_API_KEY,且该用户「游戏详情」隐私设为公开。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.steam.importer import resolve_library

router = APIRouter()


class LibraryRequest(BaseModel):
    mode: str  # steamid | url | names
    value: str


@router.post("/library")
async def import_library(req: LibraryRequest) -> dict:
    try:
        games = await resolve_library(req.mode, req.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # httpx / 解析等
        raise HTTPException(status_code=502, detail=f"读取游戏库失败: {exc}") from exc

    return {"count": len(games), "games": games}
