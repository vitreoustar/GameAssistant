import { useState } from 'react'
import type { Game } from '../types'
import { importLibrary } from '../api'
import SteamIdGuide from './SteamIdGuide'

interface Props {
  games: Game[]
  onLoaded: (games: Game[]) => void
}

export default function LibraryPanel({ games, onLoaded }: Props) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showGuide, setShowGuide] = useState(false)

  async function doImport(steamid: string) {
    setLoading(true)
    setError('')
    try {
      const gs = await importLibrary('steamid', steamid)
      onLoaded(gs)
      setShowGuide(false)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h3>
        游戏库{games.length > 0 && <span className="count">{games.length}</span>}
      </h3>

      {games.length === 0 ? (
        <button className="primary" onClick={() => setShowGuide(true)}>
          导入 Steam 游戏库
        </button>
      ) : (
        <button className="ghost" onClick={() => setShowGuide(true)}>
          ↻ 重新导入
        </button>
      )}

      {error && <div className="error">{error}</div>}

      {games.length > 0 && (
        <ul className="game-list">
          {games.slice(0, 30).map((g) => (
            <li key={g.appid}>{g.name}</li>
          ))}
          {games.length > 30 && <li className="muted">…共 {games.length} 款</li>}
        </ul>
      )}

      {showGuide && (
        <SteamIdGuide
          onImport={doImport}
          onClose={() => setShowGuide(false)}
          importing={loading}
        />
      )}
    </div>
  )
}
