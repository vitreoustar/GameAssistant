"""集中读取环境变量配置。

真实密钥只存在于 backend/.env（已被 .gitignore 忽略），
示例模板见 backend/.env.example。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# backend/ 目录（即 .env 所在位置）
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    # DeepSeek (LLM)
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

    # Steam
    steam_api_key: str = os.getenv("STEAM_API_KEY", "")
    steam_id: str = os.getenv("STEAM_ID", "")

    # Steam 网络访问(国内直连 Steam 常被拦截,可配置本地代理)
    steam_proxy: str = os.getenv("STEAM_PROXY", "")
    steam_verify_ssl: bool = os.getenv("STEAM_VERIFY_SSL", "1") != "0"

    # Steam 请求限速(全局令牌桶):rate = 每秒请求数,burst = 允许的突发令牌数
    steam_rate_limit: float = float(os.getenv("STEAM_RATE_LIMIT", "10"))
    steam_rate_burst: int = int(os.getenv("STEAM_RATE_BURST", "5"))

    # Embedding（本地 sentence-transformers）
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 前端开发服务器地址（CORS）
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


settings = Settings()

# 让 huggingface_hub / sentence-transformers 使用镜像(国内下载模型/数据集)
_hf_endpoint = os.getenv("HF_ENDPOINT", "")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint
