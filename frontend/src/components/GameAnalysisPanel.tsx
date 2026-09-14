import type { GameAnalysis } from '../types'

export default function GameAnalysisPanel({ analysis }: { analysis: GameAnalysis | null }) {
  if (!analysis) return null
  return (
    <div className="card">
      <h3>游戏评测</h3>
      <div className="rec-card">
        {analysis.header_image && (
          <img className="rec-cover" src={analysis.header_image} alt={analysis.name} loading="lazy" />
        )}
        <div className="rec-body">
          <div className="rec-name">
            {analysis.name}
            {analysis.owned && <span className="tag own-tag">已拥有</span>}
          </div>
          <div className="rec-tags">
            {analysis.genres
              .split(', ')
              .filter(Boolean)
              .slice(0, 3)
              .map((g) => (
                <span key={g} className="tag">
                  {g}
                </span>
              ))}
          </div>
          <div className="rec-meta">
            <span className={`rating ${analysis.rating.includes('好评') ? 'good' : analysis.rating.includes('差评') ? 'bad' : ''}`}>
              {analysis.rating}
            </span>
            <span>
              {analysis.price_cents != null && analysis.price_cents > 0
                ? `$${(analysis.price_cents / 100).toFixed(2)}`
                : '免费'}
            </span>
          </div>
          {analysis.match_genres.length > 0 && (
            <div className="rec-reason">✅ 与你偏好重叠:{analysis.match_genres.join('、')}</div>
          )}
        </div>
      </div>
    </div>
  )
}
