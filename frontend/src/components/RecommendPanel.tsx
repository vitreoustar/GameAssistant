import type { Candidate } from '../types'

export default function RecommendPanel({ candidates }: { candidates: Candidate[] }) {
  if (!candidates.length) {
    return (
      <div className="card">
        <h3>推荐游戏</h3>
        <p className="muted">暂无数据,在聊天里说「推荐几款游戏」</p>
      </div>
    )
  }
  return (
    <div className="card">
      <h3>推荐游戏</h3>
      <div className="rec-list">
        {candidates.map((c) => (
          <div key={c.appid} className="rec-card">
            <div className="rec-name">{c.name}</div>
            <div className="muted rec-genres">{c.genres}</div>
            <div className="rec-meta">
              ${(c.price_cents / 100).toFixed(2)} · {c.positive.toLocaleString()} 好评
              {c.metacritic_score > 0 && ` · MC ${c.metacritic_score}`}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
