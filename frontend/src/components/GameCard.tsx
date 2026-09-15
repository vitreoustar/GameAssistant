import type { GameCard } from '../types'

function ratingClass(rating: string): string {
  if (rating.includes('好评')) return 'good'
  if (rating.includes('差评')) return 'bad'
  if (rating.includes('褒贬不一')) return 'mixed'
  return ''
}

export default function GameCardView({ card }: { card: GameCard }) {
  const price =
    card.price_cents != null && card.price_cents > 0
      ? `$${(card.price_cents / 100).toFixed(2)}`
      : '免费'

  return (
    <div className="game-card">
      {card.header_image && (
        <img className="game-card-cover" src={card.header_image} alt={card.name} loading="lazy" />
      )}
      <div className="game-card-body">
        <div className="game-card-name">
          {card.name}
          {card.owned && <span className="tag own-tag">已拥有</span>}
        </div>
        <div className="game-card-tags">
          {card.genres
            .split(', ')
            .filter(Boolean)
            .slice(0, 4)
            .map((g) => (
              <span key={g} className="tag">
                {g}
              </span>
            ))}
        </div>
        <div className="game-card-meta">
          <span className={`rating ${ratingClass(card.rating)}`}>{card.rating}</span>
          <span>{price}</span>
        </div>
        {card.match_genres && card.match_genres.length > 0 && (
          <div className="game-card-match">✅ 与你偏好重叠:{card.match_genres.join('、')}</div>
        )}
        {card.reason && <div className="game-card-reason">💡 {card.reason}</div>}
      </div>
    </div>
  )
}
