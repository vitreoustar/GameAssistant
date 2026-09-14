"""共享 Pydantic schema。"""
from pydantic import BaseModel


class Game(BaseModel):
    appid: int
    name: str = ""
    playtime_forever: float | None = None
