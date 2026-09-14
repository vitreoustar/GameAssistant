import type { Candidate, Profile } from '../types'

interface Props {
  profile: Profile | null
  candidates: Candidate[]
}

export default function RecommendPanel({ profile, candidates }: Props) {
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
      {profile && profile.top_genres.length > 0 && (
        <div className="profile-box">
          <span className="muted">你的偏好</span>
          <div className="profile-tags">
            {profile.top_genres.slice(0, 5).map((g) => (
              <span key={g} className="tag">
                {g}
              </span>
            ))}
          </div>
        </div>
      )}
      <div className="rec-list">
        {candidates.map((c) => (
          <div key={c.appid} className="rec-card">
            {c.header_image && (
              <img className="rec-cover" src={c.header_image} alt={c.name} loading="lazy" />
            )}
            <div className="rec-body">
              <div className="rec-name">{c.name}</div>
              <div className="rec-tags">
                {c.genres
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
                <span className={`rating ${ratingClass(c.rating)}`}>{c.rating}</span>
                <span>{c.price_cents > 0 ? `$${(c.price_cents / 100).toFixed(2)}` : '免费'}</span>
              </div>
              {c.reason && <div className="rec-reason">💡 {c.reason}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ratingClass(rating: string): string {
  if (rating.includes('好评')) return 'good'
  if (rating.includes('差评')) return 'bad'
  if (rating.includes('褒贬不一')) return 'mixed'
  return ''
}
