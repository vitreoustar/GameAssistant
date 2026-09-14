import { useState } from 'react'
import type { Game } from '../types'
import { importLibrary } from '../api'
import { DEMO_GAMES } from '../demo'

interface Props {
  games: Game[]
  onLoaded: (games: Game[]) => void
}

const MODES = [
  { value: 'steamid', label: 'SteamID', placeholder: '7656119xxxxxxxxxx' },
  { value: 'url', label: '主页 URL', placeholder: 'https://steamcommunity.com/id/xxx' },
  { value: 'names', label: '粘贴列表', placeholder: '每行一个游戏名' },
]

export default function LibraryPanel({ games, onLoaded }: Props) {
  const [mode, setMode] = useState('names')
  const [value, setValue] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function connect() {
    setLoading(true)
    setError('')
    try {
      const gs = await importLibrary(mode, value)
      onLoaded(gs)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const placeholder = MODES.find((m) => m.value === mode)?.placeholder ?? ''

  return (
    <div className="card">
      <h3>
        游戏库{games.length > 0 && <span className="count">{games.length}</span>}
      </h3>
      <div className="mode-row">
        {MODES.map((m) => (
          <button
            key={m.value}
            className={`seg${mode === m.value ? ' active' : ''}`}
            onClick={() => setMode(m.value)}
          >
            {m.label}
          </button>
        ))}
      </div>
      {mode === 'names' ? (
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder}
          rows={5}
        />
      ) : (
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder}
        />
      )}
      <button className="primary" onClick={connect} disabled={loading || !value.trim()}>
        {loading ? '连接中…' : '连接游戏库'}
      </button>
      <button className="ghost" onClick={() => onLoaded(DEMO_GAMES)}>
        或用示例库体验(断网可演示)
      </button>
      {error && <div className="error">{error}</div>}
      {games.length > 0 && (
        <ul className="game-list">
          {games.slice(0, 30).map((g) => (
            <li key={g.appid}>{g.name}</li>
          ))}
          {games.length > 30 && <li className="muted">…共 {games.length} 款</li>}
        </ul>
      )}
    </div>
  )
}
