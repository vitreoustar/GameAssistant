"""Steam 数据层单元测试(标准库 unittest,无需额外依赖)。

运行: cd backend && .venv/Scripts/python -m unittest tests.test_steam_client -v
"""
import unittest

from app.steam.client import _norm, _parse_games_html

SAMPLE_HTML = """
<div class="games_list">
  <div class="gameListRow" id="game_1086940" data-panel="{&quot;type&quot;:&quot;PanelGroup&quot;}">
    <div class="gameListRowItemName">
      <a href="https://store.steampowered.com/app/1086940/Baldurs_Gate_3/">Baldur's Gate 3</a>
    </div>
    <div class="gameListRowItem"><h5>120.5 hrs on record</h5></div>
  </div>
  <div class="gameListRow" id="game_570">
    <div class="gameListRowItemName">
      <a href="https://store.steampowered.com/app/570/Dota_2/">Dota 2</a>
    </div>
  </div>
</div>
"""


class TestParseGamesHtml(unittest.TestCase):
    def test_extract_appid_name_playtime(self):
        games = _parse_games_html(SAMPLE_HTML)
        self.assertEqual(len(games), 2)
        self.assertEqual(games[0]["appid"], 1086940)
        self.assertEqual(games[0]["name"], "Baldur's Gate 3")
        self.assertAlmostEqual(games[0]["playtime_forever"], 120.5)
        self.assertEqual(games[1]["appid"], 570)
        self.assertEqual(games[1]["name"], "Dota 2")
        self.assertIsNone(games[1]["playtime_forever"])

    def test_empty_html_returns_empty(self):
        self.assertEqual(_parse_games_html("<html></html>"), [])


class TestNorm(unittest.TestCase):
    def test_norm_strips_punctuation(self):
        self.assertEqual(_norm("Baldur's Gate 3"), "baldurs gate 3")
        self.assertEqual(_norm("Dota 2"), "dota 2")
        self.assertEqual(_norm("  The Witcher 3: Wild Hunt  "), "the witcher 3 wild hunt")
        self.assertEqual(_norm("Overwatch® 2"), "overwatch 2")


if __name__ == "__main__":
    unittest.main()
