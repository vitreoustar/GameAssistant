"""FastAPI 入口。"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import chat, dlc, library, recommend

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时预热本地推荐索引。

    冷加载约 0.9s(索引 + 模型 + 派生结构);不预热的话这段耗时落在**第一个**
    推荐请求上。预热失败不影响服务启动 —— 推荐节点会自动退回 RAG 路径。
    """
    try:
        from app.recommend import service as recommend_service

        if recommend_service.get_recommender() is None:
            logger.warning(
                "本地推荐索引不可用,将退回 RAG 路径: %s",
                recommend_service.unavailable_reason(),
            )
        else:
            logger.info("本地推荐索引已就绪")
    except Exception as exc:  # noqa: BLE001 - 预热失败不应阻止启动
        logger.warning("推荐索引预热失败: %s", exc)
    yield


app = FastAPI(title="SteamAssistant", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(library.router, prefix="/api")
app.include_router(dlc.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "SteamAssistant", "docs": "/docs"}
