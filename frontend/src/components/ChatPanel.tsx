import { useState } from 'react'
import type { ChatMessage, Step } from '../types'

interface Props {
  messages: ChatMessage[]
  steps: Step[]
  streaming: boolean
  onSend: (text: string) => void
}

export default function ChatPanel({ messages, steps, streaming, onSend }: Props) {
  const [input, setInput] = useState('')
  const active = steps.filter((s) => s.status === 'start').map((s) => s.node)

  function send(text?: string) {
    const t = (text ?? input).trim()
    if (!t || streaming) return
    setInput('')
    onSend(t)
  }

  return (
    <div className="chat-panel">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            先连接右侧游戏库,然后问我「整理我的 DLC」或「推荐几款我喜欢的游戏」
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">{m.content}</div>
          </div>
        ))}
        {streaming && active.length > 0 && (
          <div className="steps">
            {active.map((n) => (
              <span key={n} className="step-pill">
                ⏳ {n}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="chat-input">
        <div className="quick-btns">
          <button onClick={() => send('整理一下我的 DLC')}>📦 整理 DLC</button>
          <button onClick={() => send('推荐几款我可能喜欢的游戏')}>🎮 推荐游戏</button>
        </div>
        <div className="input-row">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') send()
            }}
            placeholder="输入消息…"
          />
          <button className="primary" onClick={() => send()} disabled={streaming}>
            {streaming ? '生成中…' : '发送'}
          </button>
        </div>
      </div>
    </div>
  )
}
