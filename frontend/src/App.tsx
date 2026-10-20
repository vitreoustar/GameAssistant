import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import ChatPanel from './components/ChatPanel'
import DlcPanel from './components/DlcPanel'
import LibraryPanel from './components/LibraryPanel'
import { streamChat } from './api'
import type { ChatMessage, ChatTurn, DlcReport, Game, GameCard, RecommendState, Step } from './types'

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

function setLastAssistant(
  ms: ChatMessage[],
  text: string,
  cards: GameCard[],
  cardPosition: 'before' | 'after',
): ChatMessage[] {
  const copy = [...ms]
  const last = copy[copy.length - 1]
  if (last && last.role === 'assistant') {
    copy[copy.length - 1] = { ...last, content: text, cards, cardPosition }
  } else {
    copy.push({ role: 'assistant', content: text, cards, cardPosition })
  }
  return copy
}

type CardLike = {
  appid: number
  name: string
  header_image?: string
  genres?: string
  price_cents?: number | null
  rating?: string
  reason?: string
  match_genres?: string[]
  pref_match?: string[]
  owned?: boolean
}

// 把各种结构化结果统一转成卡片
function toCard(c: CardLike): GameCard {
  return {
    appid: c.appid,
    name: c.name,
    header_image: c.header_image || '',
    genres: c.genres || '',
    price_cents: c.price_cents ?? null,
    rating: c.rating || '',
    reason: c.reason,
    match_genres: c.match_genres,
    pref_match: c.pref_match,
    owned: c.owned,
  }
}

const HISTORY_TURNS = 8
const HISTORY_CHARS = 300

export default function App() {
  const [library, setLibrary] = useState<Game[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [steps, setSteps] = useState<Step[]>([])
  const [streaming, setStreaming] = useState(false)
  const [dlcReport, setDlcReport] = useState<DlcReport | null>(null)
  const [lastCandidates, setLastCandidates] = useState<{ appid: number; name: string }[]>([])
  // 多轮推荐状态(偏好 + 已展示的 appid)。后端在 done 里给出,这里原样回传。
  const [recommendState, setRecommendState] = useState<RecommendState | null>(null)

  // 历史对话只用于让路由模型理解「好」「换一批」这类省略,不进 LLM 的主上下文。
  // 用 ref 而不是 state:send 的闭包会过期,而 ref 在调用瞬间读到的是**本次发送之前**的消息。
  const historyRef = useRef<ChatTurn[]>([])
  useEffect(() => {
    historyRef.current = messages
      .filter((m) => m.content.trim())
      .slice(-HISTORY_TURNS)
      .map((m) => ({ role: m.role, text: m.content.slice(0, HISTORY_CHARS) }))
  }, [messages])

  const send = useCallback(
    async (text: string) => {
      if (streaming) return
      const history = historyRef.current
      setMessages((m) => [
        ...m,
        { role: 'user', content: text },
        { role: 'assistant', content: '' },
      ])
      setSteps([])
      setDlcReport(null)
      setStreaming(true)
      try {
        await streamChat(
          text,
          library,
          { candidates: lastCandidates, recommend: recommendState, history },
          (ev) => {
            if (ev.type === 'node') {
              setSteps((s) =>
                ev.data.status === 'start'
                  ? [...s, ev.data]
                  : s.map((x) => (x.node === ev.data.node ? { ...x, status: 'end' } : x)),
              )
            } else if (ev.type === 'token') {
              setMessages((m) => appendAssistant(m, ev.data))
            } else if (ev.type === 'done') {
              const d = ev.data
              // ⚠️ 必须无条件更新:即使这一轮没推荐任何东西,状态里的 shown 也可能变了
              if (d.recommend_state) setRecommendState(d.recommend_state)
              let cards: GameCard[] = []
              let pos: 'before' | 'after' = 'after'
              if (d.intent === 'recommend' && d.candidates) {
                cards = d.candidates.map(toCard)
                pos = 'after'
                setLastCandidates(d.candidates.map((c) => ({ appid: c.appid, name: c.name })))
              } else if (d.game_analysis) {
                cards = [toCard(d.game_analysis)]
                pos = 'before'
              } else if (d.game_candidates) {
                cards = d.game_candidates.map(toCard)
                pos = 'after'
                setLastCandidates(d.game_candidates.map((c) => ({ appid: c.appid, name: c.name })))
              }
              setMessages((m) => setLastAssistant(m, d.text, cards, pos))
              setDlcReport(d.dlc_report)
            } else if (ev.type === 'error') {
              setMessages((m) => appendAssistant(m, '⚠️ ' + ev.data))
            }
          },
        )
      } catch (e) {
        setMessages((m) => setLastAssistant(m, '⚠️ 请求失败: ' + (e as Error).message, [], 'after'))
      } finally {
        setStreaming(false)
      }
    },
    [library, streaming, lastCandidates, recommendState],
  )

  // 换了游戏库就重置多轮状态:旧的 `shown` 是上一份库的推荐记录,
  // 约束留着会让新库一上来就没东西可推。
  const handleLibraryLoaded = useCallback((games: Game[]) => {
    setLibrary(games)
    setRecommendState(null)
    setLastCandidates([])
  }, [])

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
          <LibraryPanel games={library} onLoaded={handleLibraryLoaded} />
          <DlcPanel report={dlcReport} />
        </aside>
      </main>
    </div>
  )
}
