import { useState } from 'react'

interface Props {
  onImport: (steamid: string) => void
  onClose: () => void
  importing: boolean
}

const STEAMID_RE = /^\d{17}$/

export default function SteamIdGuide({ onImport, onClose, importing }: Props) {
  const [steamid, setSteamid] = useState('')
  const valid = STEAMID_RE.test(steamid.trim())

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>获取你的 SteamID</h3>
          <button className="close" onClick={onClose} aria-label="关闭">
            ×
          </button>
        </div>

        <ol className="guide">
          <li>点击下方按钮打开 Steam 账号页,并登录你的 Steam 账号</li>
          <li>
            登录后页面顶部会显示 <b>Steam ID: 7656…</b> 这串 17 位数字
          </li>
          <li>
            确认 Steam 隐私设置里「我的个人资料」与「游戏详情」设为<b>公开</b>
          </li>
          <li>把 SteamID 粘贴到下面并导入</li>
        </ol>

        <a
          className="primary link"
          href="https://store.steampowered.com/account/"
          target="_blank"
          rel="noreferrer"
        >
          打开 Steam 账号页 ↗
        </a>

        <input
          value={steamid}
          onChange={(e) => setSteamid(e.target.value)}
          placeholder="7656119xxxxxxxxxx"
          inputMode="numeric"
        />
        <button
          className="primary"
          disabled={importing || !valid}
          onClick={() => onImport(steamid.trim())}
        >
          {importing ? '导入中…' : '导入游戏库'}
        </button>
      </div>
    </div>
  )
}
