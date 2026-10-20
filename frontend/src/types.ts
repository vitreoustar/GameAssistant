export interface Game {
  appid: number
  name: string
  playtime_forever: number | null
}

export interface DlcItem {
  appid: number
  name: string
  owned: boolean
  price_cents: number | null
  currency: string | null
}

export interface DlcRow {
  appid: number
  name: string
  owned_dlc: DlcItem[]
  missing_dlc: DlcItem[]
  missing_count: number
  owned_count: number
  truncated: boolean
}

export interface DlcReport {
  games_with_dlc: number
  total_missing_dlc: number
  total_missing_price_cents: number
  processed_games: number
  total_games: number
  capped: boolean
  rows: DlcRow[]
}

export interface Profile {
  top_genres: string[]
  top_categories: string[]
  summary: string
}

export interface Candidate {
  appid: number
  name: string
  header_image: string
  genres: string
  categories: string
  price_cents: number
  positive: number
  negative: number
  rating: string
  metacritic_score: number
  distance: number
  reason?: string
  pref_match?: string[]
}

// 多轮推荐的对话状态:偏好 + 已展示过的 appid。
// 由后端在 done 事件里给出,前端原样回传 —— 服务端不存会话。
export interface RecommendState {
  genres: string[]
  avoid: string[]
  unmapped: string[]
  shown: number[]
  turn: number
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  text: string
}

export interface GameAnalysis {
  appid: number
  name: string
  header_image: string
  genres: string
  categories: string
  price_cents: number | null
  positive: number
  negative: number
  rating: string
  metacritic_score: number
  match_genres: string[]
  owned: boolean
}

// 对话内统一的卡片结构
export interface GameCard {
  appid: number
  name: string
  header_image: string
  genres: string
  price_cents: number | null
  rating: string
  reason?: string
  match_genres?: string[]
  pref_match?: string[]
  owned?: boolean
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  cards?: GameCard[]
  cardPosition?: 'before' | 'after'
}

export interface Step {
  node: string
  status: 'start' | 'end'
}

export type StreamEvent =
  | { type: 'node'; data: { node: string; status: 'start' | 'end' } }
  | { type: 'token'; node: string; data: string }
  | {
      type: 'done'
      data: {
        text: string
        intent: string | null
        dlc_report: DlcReport | null
        profile: Profile | null
        candidates: Candidate[] | null
        game_analysis: GameAnalysis | null
        game_candidates: GameCard[] | null
        recommend_state: RecommendState | null
      }
    }
  | { type: 'error'; data: string }
