import { useCallback, useState } from 'react'
import './App.css'
import ChatPanel from './components/ChatPanel'
import DlcPanel from './components/DlcPanel'
import LibraryPanel from './components/LibraryPanel'
import RecommendPanel from './components/RecommendPanel'
import { streamChat } from './api'
import type { Candidate, ChatMessage, DlcReport, Game, Step } from './types'

function appendAssistant(ms: ChatMessage[], text: string): ChatMessage[] {
  const copy = [...ms]
  const last = copy[copy.length - 1]
  if (last && last.role === 'assistant') {
    copy[copy.length - 1] = { ...last, content: last.content + text }
  } else {
    copy.push({ role: 'assistant', content: text })
  }
  return copy
}

function setAssistant(ms: ChatMessage[], text: string): ChatMessage[] {
  const copy = [...ms]
  const last = copy[copy.length - 1]
  if (last && last.role === 'assistant') {
    copy[copy.length - 1] = { ...last, content: text }
  } else {
    copy.push({ role: 'assistant', content: text })
  }
  return copy
}

export default function App() {
  const [library, setLibrary] = useState<Game[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [steps, setSteps] = useState<Step[]>([])
  const [streaming, setStreaming] = useState(false)
  const [dlcReport, setDlcReport] = useState<DlcReport | null>(null)
  const [candidates, setCandidates] = useState<Candidate[]>([])

  const send = useCallback(
    async (text: string) => {
      if (streaming) return
      setMessages((m) => [
        ...m,
        { role: 'user', content: text },
        { role: 'assistant', content: '' },
      ])
      setSteps([])
      setStreaming(true)
      try {
        await streamChat(text, library, (ev) => {
          if (ev.type === 'node') {
            setSteps((s) =>
              ev.data.status === 'start'
                ? [...s, ev.data]
                : s.map((x) => (x.node === ev.data.node ? { ...x, status: 'end' } : x)),
            )
          } else if (ev.type === 'token') {
            setMessages((m) => appendAssistant(m, ev.data))
          } else if (ev.type === 'done') {
            setMessages((m) => setAssistant(m, ev.data.text))
            setDlcReport(ev.data.dlc_report)
            setCandidates(ev.data.candidates || [])
          } else if (ev.type === 'error') {
            setMessages((m) => setAssistant(m, '⚠️ ' + ev.data))
          }
        })
      } catch (e) {
        setMessages((m) => setAssistant(m, '⚠️ 请求失败: ' + (e as Error).message))
      } finally {
        setStreaming(false)
      }
    },
    [library, streaming],
  )

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎮 SteamAssistant</h1>
        <span className="badge">
          {library.length ? `已连接 ${library.length} 款游戏` : '未连接游戏库'}
        </span>
      </header>
      <main className="app-body">
        <ChatPanel messages={messages} steps={steps} streaming={streaming} onSend={send} />
        <aside className="side-panel">
          <LibraryPanel games={library} onLoaded={setLibrary} />
          <DlcPanel report={dlcReport} />
          <RecommendPanel candidates={candidates} />
        </aside>
      </main>
    </div>
  )
}
