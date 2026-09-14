"""游戏库相关端点。支持三种导入模式:
  1. steamid64 + Steamworks key -> GetOwnedGames
  2. 个人主页 URL -> 抓取 games 页
  3. 粘贴游戏名列表 -> 名称匹配 appid
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
