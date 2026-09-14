import type { DlcReport } from '../types'

export default function DlcPanel({ report }: { report: DlcReport | null }) {
  if (!report) {
    return (
      <div className="card">
        <h3>DLC 报告</h3>
        <p className="muted">暂无数据,在聊天里说「整理我的 DLC」</p>
      </div>
    )
  }
  return (
    <div className="card">
      <h3>DLC 报告</h3>
      {report.capped && (
        <p className="muted">
          仅分析了你最常玩的 {report.processed_games}/{report.total_games} 款游戏
        </p>
      )}
      <p className="summary">
        共缺 <b>{report.total_missing_dlc}</b> 个 DLC
        {report.total_missing_price_cents > 0 && (
          <>
            {' '}· 已列出合计{' '}
            <b>${(report.total_missing_price_cents / 100).toFixed(2)}</b>
          </>
        )}
      </p>
      <div className="dlc-rows">
        {report.rows.map((r) => (
          <div key={r.appid} className="dlc-row">
            <div className="dlc-row-head">
              {r.name} <span className="muted">缺 {r.missing_count}</span>
            </div>
            {r.missing_dlc.map((d) => (
              <div key={d.appid} className="dlc-item miss">
                <span>{d.name}</span>
                {d.price_cents != null && <b>${(d.price_cents / 100).toFixed(2)}</b>}
              </div>
            ))}
            {r.truncated && (
              <div className="muted">
                …还有 {r.missing_count - r.missing_dlc.length} 个未显示,追问可展开
              </div>
            )}
            {r.owned_count > 0 && (
              <div className="dlc-item own">✓ 已拥有 {r.owned_count} 个 DLC</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
