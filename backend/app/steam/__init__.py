"""Steam 数据层：API 客户端、主页抓取、DLC 计算、导入匹配。"""
from app.steam.client import SteamClient

# 进程内共享客户端(缓存/连接复用)
client = SteamClient()
