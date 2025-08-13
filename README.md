# Market Pulse Agent

Discord agent that listens to market-pulse posts, computes Opening Range (OR) levels with live price checks, and enriches with RSI, MACD, ATR, earnings proximity, and 14–21D option ideas. Schedules summaries (every 30m early, hourly later) and posts a concise bias with entry/stop targets. Config via `.env` + `config.yaml`.

## Features
- Reads market-pulse messages from a source channel
- Classifies tickers by OR status (Above OR / Testing OR / Weak / Extremely Weak)
- Adds RSI, MACD (hist), ATR, earnings-in-7d, and best 14–21D options by volume
- Scheduled summaries during market hours (30m early, hourly later)
- Fully configurable via `.env` and `config.yaml`

## Requirements
- Python 3.10+
- A Discord bot (token from the Discord Developer Portal)

Install deps:
```bash
pip install -r requirements.txt
```

## Configure
1. Create `.env` (never commit this):
```
DISCORD_BOT_TOKEN=YOUR_DISCORD_BOT_TOKEN
READ_CHANNEL_ID=1401965318414536800
POST_CHANNEL_ID=1402142866498654349
```
2. (Optional) Edit `config.yaml` to change timezone, market hours, schedule, and default tickers.

## Run
```bash
python market_pulse_agent.py
```

The agent posts summaries to `POST_CHANNEL_ID` when it detects a market-pulse block or when the scheduler fires during market hours.

## Security
- Keep secrets in `.env` only (the file is gitignored).
- Do not hard-code tokens or channel IDs in code.

## License
MIT — see `LICENSE`.