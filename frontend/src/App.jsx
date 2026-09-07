import { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "https://ufc-fight-predictor-ep0g.onrender.com";

function getInitials(name) {
  if (!name) return "?";
  return name
    .split(" ")
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

function weightClassWarning(stats1, stats2) {
  if (!stats1 || !stats2) return null;
  const diff = Math.abs(stats1.weight_lbs - stats2.weight_lbs);
  if (diff > 20) {
    return "These fighters are in different weight classes — predictions across large weight gaps are less reliable, since real fights rarely happen at this mismatch.";
  }
  return null;
}

function FighterCard({ label, name, fighters, onChange, stats, isWinner, fightersLoading }) {
  return (
    <div className={`fighter-card ${isWinner ? "fighter-card--winner" : ""}`}>
      <span className="fighter-card__label">{label}</span>
      <div className="fighter-card__avatar">{getInitials(name)}</div>
      <select
        className="fighter-card__select"
        value={name}
        onChange={(e) => onChange(e.target.value)}
        disabled={fightersLoading}
      >
        <option value="">
          {fightersLoading ? "Waking up server… (~30-60s)" : "Choose a fighter"}
        </option>
        {fighters.map((f) => (
          <option key={f} value={f}>{f}</option>
        ))}
      </select>
      {stats && (
        <dl className="fighter-card__stats">
          <div><dt>Height</dt><dd>{stats.height}"</dd></div>
          <div><dt>Weight</dt><dd>{stats.weight_lbs} lbs</dd></div>
          <div><dt>Reach</dt><dd>{stats.reach_inches}"</dd></div>
          <div><dt>Age</dt><dd>{Math.round(stats.age)}</dd></div>
          <div><dt>Strikes/min</dt><dd>{stats.slpm}</dd></div>
          <div><dt>Takedowns/15min</dt><dd>{stats.td_avg}</dd></div>
        </dl>
      )}
    </div>
  );
}

export default function App() {
  const [fighters, setFighters] = useState([]);
  const [fightersLoading, setFightersLoading] = useState(true);
  const [f1, setF1] = useState("");
  const [f2, setF2] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [slowLoad, setSlowLoad] = useState(false);
  const weightWarning = result?.stat_comparison?.weight_lbs
    ? weightClassWarning(
        { weight_lbs: result.stat_comparison.weight_lbs.fighter_1 },
        { weight_lbs: result.stat_comparison.weight_lbs.fighter_2 }
      )
    : null;

  useEffect(() => {
    fetch(`${API_BASE}/fighters`)
      .then((res) => res.json())
      .then((data) => setFighters(data.fighters))
      .catch(() => setError("Couldn't reach the prediction server. Is the backend running?"))
      .finally(() => setFightersLoading(false));
  }, []);

  const canPredict = f1 && f2 && f1 !== f2;

  async function handlePredict() {
    setLoading(true);
    setError("");
    setResult(null);
    const slowTimer = setTimeout(() => setSlowLoad(true), 4000);
    try {
      const res = await fetch(
        `${API_BASE}/predict?f1=${encodeURIComponent(f1)}&f2=${encodeURIComponent(f2)}`
      );
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.detail || "Prediction failed");
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      clearTimeout(slowTimer);
      setLoading(false);
      setSlowLoad(false);
    }
  }

  const f1Wins = result && result.fighter_1_win_probability > result.fighter_2_win_probability;
  const f2Wins = result && !f1Wins;

  return (
    <div className="app">
      <header className="app__header">
        <h1>UFC Predictor</h1>
        <p>Pick two fighters. See who the numbers favor.</p>
      </header>

      <div className="matchup">
        <FighterCard
          label="Corner 1"
          name={f1}
          fighters={fighters}
          onChange={setF1}
          stats={result?.stat_comparison ? {
            height: result.stat_comparison.height?.fighter_1,
            weight_lbs: result.stat_comparison.weight_lbs?.fighter_1,
            reach_inches: result.stat_comparison.reach_inches?.fighter_1,
            age: result.stat_comparison.age?.fighter_1,
            slpm: result.stat_comparison.slpm?.fighter_1,
            td_avg: result.stat_comparison.td_avg?.fighter_1,
          } : null}
          isWinner={f1Wins}
          fightersLoading={fightersLoading}
        />

        <div className="matchup__center">
          <span className="matchup__vs">VS</span>
          <button
            className="matchup__predict-btn"
            onClick={handlePredict}
            disabled={!canPredict || loading}
          >
            {loading ? (slowLoad ? "Waking up server…" : "Calculating…") : "Predict"}
          </button>
        </div>

        <FighterCard
          label="Corner 2"
          name={f2}
          fighters={fighters}
          onChange={setF2}
          stats={result?.stat_comparison ? {
            height: result.stat_comparison.height?.fighter_2,
            weight_lbs: result.stat_comparison.weight_lbs?.fighter_2,
            reach_inches: result.stat_comparison.reach_inches?.fighter_2,
            age: result.stat_comparison.age?.fighter_2,
            slpm: result.stat_comparison.slpm?.fighter_2,
            td_avg: result.stat_comparison.td_avg?.fighter_2,
          } : null}
          isWinner={f2Wins}
          fightersLoading={fightersLoading}
        />
      </div>

      {error && <p className="app__error">{error}</p>}
      
      {weightWarning && <p className="app__warning">{weightWarning}</p>}

      {result && (
        <div className="probability">
          <div className="probability__bar">
            <div
              className="probability__fill probability__fill--f1"
              style={{ width: `${result.fighter_1_win_probability * 100}%` }}
            />
          </div>
          <div className="probability__labels">
            <span>{result.fighter_1}: {(result.fighter_1_win_probability * 100).toFixed(1)}%</span>
            <span>{result.fighter_2}: {(result.fighter_2_win_probability * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}