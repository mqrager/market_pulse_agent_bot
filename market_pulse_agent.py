# market_pulse_agent_patched.py
import os, sys, time, re, asyncio
from datetime import datetime, date, timedelta, time as dtime
from pathlib import Path

import pandas as pd
import discord
import pytz
import yfinance as yf
import yaml
from dotenv import load_dotenv

# =========================
# Load .env and config.yaml
# =========================
SCRIPT_NAME = os.path.basename(__file__)

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _print_banner():
    print("=" * 70)
    print(f"🚀 Starting script: {SCRIPT_NAME}")
    print("=" * 70)

def load_config(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

# Always load .env from the script folder (works no matter the launch dir)
ENV_PATH = Path(__file__).with_name(".env")
CONFIG_PATH = Path(__file__).with_name("config.yaml")

_print_banner()
print(f"[{ts()}] ({SCRIPT_NAME}) Loading .env from: {ENV_PATH}")
loaded = load_dotenv(dotenv_path=ENV_PATH)
print(f"[{ts()}] ({SCRIPT_NAME}) .env loaded: {loaded}")

CFG = load_config(CONFIG_PATH)

def cfg(path, default=None):
    """Mini dotted-path getter for YAML (e.g., cfg('market.update_schedule.early_interval_minutes'))."""
    cur = CFG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# =========================
# Settings from config.yaml
# =========================
# Secrets from .env
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not BOT_TOKEN:
    sys.exit("[ERROR] DISCORD_BOT_TOKEN is not set in .env")

# Channels: prefer .env overrides if present, else config.yaml
READ_CHANNEL_ID  = int(os.getenv("READ_CHANNEL_ID",  cfg("discord.read_channel_id",  "0")))
WRITE_CHANNEL_ID = int(os.getenv("POST_CHANNEL_ID",  cfg("discord.write_channel_id", "0")))

# Timezones & market hours
LOCAL_TZ_NAME = cfg("market.timezone", "US/Pacific")
TIMEZONE      = pytz.timezone(LOCAL_TZ_NAME)
MARKET_TZ     = pytz.timezone("US/Eastern")

MARKET_OPEN_STR  = cfg("market.market_open",  "09:30")
MARKET_CLOSE_STR = cfg("market.market_close", "16:00")
# Parse "HH:MM"
mo_h, mo_m = map(int, MARKET_OPEN_STR.split(":"))
mc_h, mc_m = map(int, MARKET_CLOSE_STR.split(":"))
MARKET_OPEN  = dtime(mo_h, mo_m)
MARKET_CLOSE = dtime(mc_h, mc_m)

# Schedule intervals
EARLY_MINUTES    = int(cfg("market.update_schedule.early_duration_minutes", 150))   # first 2.5h
EARLY_INTERVAL   = int(cfg("market.update_schedule.early_interval_minutes", 30))
LATER_INTERVAL   = int(cfg("market.update_schedule.later_interval_minutes", 60))

# Tickers (from YAML)
DEFAULT_TICKERS = cfg("market.tickers", ["SPY","QQQ","DIA"])

# Logging verbosity & previews
LOG_LEVEL      = (cfg("log.level", "INFO") or "INFO").upper()
HEARTBEAT_SECS = int(cfg("log.heartbeat_secs", 120))
SAMPLE_ROWS    = int(cfg("log.sample_rows", 3))

# ===========
# Logging API
# ===========
def log(level, msg):
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    if level not in levels: level = "INFO"
    if levels.index(level) >= levels.index(LOG_LEVEL):
        print(f"[{ts()}] ({SCRIPT_NAME}) [{level}] {msg}")

def debug(msg): log("DEBUG", msg)
def info(msg):  log("INFO",  msg)
def warn(msg):  log("WARN",  msg)
def error(msg): log("ERROR", msg)

info(f"READ_CHANNEL_ID={READ_CHANNEL_ID}, WRITE_CHANNEL_ID={WRITE_CHANNEL_ID}, LocalTZ={LOCAL_TZ_NAME}, MarketTZ=US/Eastern")
info(f"Schedule: early {EARLY_INTERVAL}m for {EARLY_MINUTES}m after open, then {LATER_INTERVAL}m until close")
info(f"Tickers: {', '.join(DEFAULT_TICKERS)}")

# ============================
# Discord intents & client
# ============================
intents = discord.Intents.default()
intents.message_content = True

# ============================
# Helpers & indicator functions
# ============================
def _preview_df(df, name):
    if df is None or getattr(df, "empty", True):
        warn(f"{name}: empty dataframe")
        return
    rows = min(SAMPLE_ROWS, len(df))
    info(f"{name}: rows={len(df)}; preview top {rows}:")
    try:
        print(df.head(rows).to_string())
    except Exception:
        print(str(df.head(rows)))

def fetch_rsi(ticker):
    try:
        info(f"[API] yfinance.download({ticker}, period='7d', interval='15m')")
        df = yf.download(ticker, period="7d", interval="15m")
        _preview_df(df, f"{ticker} 15m/7d")
        if df.empty:
            raise ValueError("Empty dataframe from yfinance (15m)")
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = round(float(rsi.dropna().iloc[-1]), 2)
        info(f"{ticker} RSI(14) = {latest_rsi}")
        return latest_rsi
    except Exception as e:
        error(f"[RSI ERROR] {ticker}: {e}")
        return "N/A"

def fetch_additional_indicators(ticker):
    try:
        info(f"[API] yfinance.download({ticker}, period='60d', interval='1d')")
        df = yf.download(ticker, period="60d", interval="1d")
        _preview_df(df, f"{ticker} 1d/60d")
        if df is None or df.empty:
            raise ValueError("Empty dataframe from yfinance (daily)")

        df = df.dropna(how="any")
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        prev_close = close.shift(1)

        # True Range components (guard NaNs)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low  - prev_close).abs()
        tr = tr1.combine(tr2, max).combine(tr3, max)

        atr_series = tr.rolling(window=14, min_periods=14).mean()
        atr_val = atr_series.iloc[-1]
        if pd.isna(atr_val):
            raise ValueError("ATR calc returned NaN")

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal

        atr = round(float(atr_val), 2)
        macd_value = round(float(macd_hist.iloc[-1]), 2)

        info(f"{ticker} ATR(14)={atr}, MACD(hist)={macd_value}")
        return {"ATR": atr, "MACD": macd_value}
    except Exception as e:
        error(f"[INDICATOR ERROR] {ticker}: {e}")
        return {"ATR": "N/A", "MACD": "N/A"}

def fetch_earnings_date(ticker):
    try:
        info(f"[API] yfinance.Ticker({ticker}).calendar")
        cal = yf.Ticker(ticker).calendar
        try:
            _preview_df(cal, f"{ticker} calendar")
        except Exception:
            pass
        if getattr(cal, "empty", True):
            return ""
        earnings = cal.loc['Earnings Date'][0]
        today = date.today()
        if isinstance(earnings, datetime):
            earnings = earnings.date()
        days = (earnings - today).days
        note = f"⚠️ Earnings in {days} day(s): {earnings}"
        info(f"{ticker} {note}")
        return note if 0 <= days <= 7 else ""
    except Exception as e:
        warn(f"[EARNINGS ERROR] {ticker}: {e}")
        return ""

def fetch_best_option_from_yahoo(ticker):
    try:
        info(f"[API] yfinance.Ticker({ticker}).options (filter 14–21D)")
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            warn(f"{ticker} no options expiries found")
            return "No options data found."

        info(f"{ticker} expiries sample: {expiries[:3]}")
        now = datetime.now()
        valid = []
        for exp in expiries:
            try:
                d = datetime.strptime(exp, "%Y-%m-%d")
                days = (d - now).days
                if 14 <= days <= 21:
                    valid.append((exp, days))
            except Exception:
                continue

        if not valid:
            return "No suitable expiration dates in 14–21 day range."
        exp = sorted(valid, key=lambda x: x[1])[0][0]

        chain = tk.option_chain(exp)
        calls = chain.calls.sort_values(by="volume", ascending=False)
        puts  = chain.puts.sort_values(by="volume",  ascending=False)

        _preview_df(calls, f"{ticker} calls @ {exp}")
        _preview_df(puts,  f"{ticker} puts  @ {exp}")

        top_call = calls.iloc[0] if not calls.empty else None
        top_put  = puts.iloc[0]  if not puts.empty  else None

        if top_call is not None:
            info(f"{ticker} Top CALL {exp}: strike={top_call.strike}, last={top_call.lastPrice}, vol={top_call.volume}, OI={top_call.openInterest}")
        if top_put is not None:
            info(f"{ticker} Top PUT  {exp}: strike={top_put.strike}, last={top_put.lastPrice}, vol={top_put.volume}, OI={top_put.openInterest}")

        return (
            f"\n📊 **Top Option Ideas ({exp})**\n"
            + (f"CALL: Strike ${getattr(top_call,'strike','?')}, Last ${getattr(top_call,'lastPrice','?')}, Vol {getattr(top_call,'volume','?')}, OI {getattr(top_call,'openInterest','?')}\n" if top_call is not None else "CALL: n/a\n")
            + (f"PUT: Strike ${getattr(top_put,'strike','?')}, Last ${getattr(top_put,'lastPrice','?')}, Vol {getattr(top_put,'volume','?')}, OI {getattr(top_put,'openInterest','?')}" if top_put is not None else "PUT: n/a")
        )
    except Exception as e:
        error(f"[YFINANCE OPTION ERROR] {ticker}: {e}")
        return f"[YFINANCE OPTION ERROR] {ticker}: {e}"

# ============================
# OR levels & live price utils
# ============================
def compute_opening_range_30m(ticker):
    """
    Compute ORH/ORL from 09:30–10:00 ET using today's 1m data.
    Returns (or_high, or_low) or (None, None) if unavailable.
    """
    try:
        info(f"[API] yfinance.download({ticker}, period='1d', interval='1m') for OR")
        df = yf.download(ticker, period="1d", interval="1m")
        if df is None or df.empty:
            warn(f"{ticker} OR: empty 1m data")
            return None, None
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.UTC).tz_convert(MARKET_TZ)
        else:
            df.index = df.index.tz_convert(MARKET_TZ)
        start = df.index.normalize()[0] + timedelta(hours=9, minutes=30)
        end   = df.index.normalize()[0] + timedelta(hours=10, minutes=0)
        or_window = df[(df.index >= start) & (df.index <= end)]
        _preview_df(or_window, f"{ticker} 1m OR window")
        if or_window.empty:
            warn(f"{ticker} OR window empty (market closed or too early)")
            return None, None
        or_high = float(or_window["High"].max())
        or_low  = float(or_window["Low"].min())
        info(f"{ticker} ORH={or_high}, ORL={or_low}")
        return or_high, or_low
    except Exception as e:
        warn(f"{ticker} OR compute error: {e}")
        return None, None

def get_last_price(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m")
        if df is None or df.empty:
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize(pytz.UTC).tz_convert(MARKET_TZ)
        else:
            df.index = df.index.tz_convert(MARKET_TZ)
        return float(df["Close"].iloc[-1])
    except Exception as e:
        warn(f"{ticker} last price error: {e}")
        return None

# ============================
# Parsing + classification
# ============================
def classify_ticker_block(text):
    lines = text.strip().split("\n")
    data = {}
    current_ticker = ""

    for line in lines:
        match = re.match(r"^([A-Z]{2,6}) status at .*?:$", line.strip(), flags=re.I)
        if match:
            current_ticker = match.group(1).upper()
            data[current_ticker] = {"raw": [line], "OR_high": None, "OR_low": None}
        elif current_ticker:
            data[current_ticker]["raw"].append(line)
            if "opening range: high" in line.lower():
                try:
                    high = float(re.search(r"High [`$]?([\d.]+)`?", line, flags=re.I).group(1))
                    low  = float(re.search(r"Low [`$]?([\d.]+)`?",  line, flags=re.I).group(1))
                    data[current_ticker]["OR_high"] = high
                    data[current_ticker]["OR_low"]  = low
                except Exception as e:
                    warn(f"Failed to parse OR for {current_ticker}: {e}")

    summary = {"🚨 Extremely Weak": [], "⚠️ Weak": [], "🙀 Testing OR": [], "✅ Above OR": []}

    # Live price sanity-check
    for ticker, content in data.items():
        raw_text = "\n".join(content["raw"]).lower()
        h = content["OR_high"]; l = content["OR_low"]
        px = get_last_price(ticker)
        debug(f"{ticker} ORH={h} ORL={l} last={px}")

        if px is not None and h and l:
            if px >= h:
                summary["✅ Above OR"].append((ticker, content)); continue
            if px <= l and ("-3.618 sd" in raw_text or "extremely weak" in raw_text):
                summary["🚨 Extremely Weak"].append((ticker, content)); continue
            if l < px < h:
                summary["🙀 Testing OR"].append((ticker, content)); continue

        # Fallback: text-only
        if "-3.618 sd" in raw_text or "extremely weak" in raw_text:
            summary["🚨 Extremely Weak"].append((ticker, content))
        elif "-1.618 sd" in raw_text and "-3.618 sd" not in raw_text:
            summary["⚠️ Weak"].append((ticker, content))
        elif "testing or" in raw_text or "between or" in raw_text:
            summary["🙀 Testing OR"].append((ticker, content))
        elif "above or" in raw_text:
            summary["✅ Above OR"].append((ticker, content))

    return summary

def build_summary_message(summary):
    blocks = ["📊 **Market Pulse Summary** – Realtime Update\n"]
    for label, ticker_data in summary.items():
        if ticker_data:
            tickers = [f"**{ticker}**" for ticker, _ in ticker_data]
            blocks.append(f"**{label}**\n" + " • ".join(tickers) + "\n")
    return "\n".join(blocks)

def generate_insight(summary):
    num_extreme = len(summary["🚨 Extremely Weak"])
    num_weak    = len(summary["⚠️ Weak"])
    num_above   = len(summary["✅ Above OR"])

    insight_lines = ["---", "🧠 **Insight & Bias**"]

    if num_extreme >= 4:
        insight_lines.append("Market shows **heavy weakness** — especially in indices or large caps.")
    elif num_weak >= 4:
        insight_lines.append("Sustained pressure across multiple names — **risk-off environment**.")
    elif num_above >= 4:
        insight_lines.append("Multiple names holding above OR — **rotation into strength possible**.")
    else:
        insight_lines.append("Mixed conditions — no clear trend.")

    if num_extreme >= 3:
        insight_lines.append("📌 **Bias**: Favor **PUTs** on weak names, scalp only on strength.")
    elif num_above > num_extreme:
        insight_lines.append("📌 **Bias**: Consider **CALLs** on names holding above OR, watch for confirmation.")
    else:
        insight_lines.append("📌 **Bias**: Stay cautious, wait for breakouts or reclaim of OR.")

    insight_lines.append("\n💰 **Price Targets & Entry Ideas**")
    for label, ticker_data in summary.items():
        for ticker, content in ticker_data:
            or_high = content.get("OR_high")
            or_low  = content.get("OR_low")
            rsi = fetch_rsi(ticker)
            indicators = fetch_additional_indicators(ticker)
            macd = indicators.get("MACD", "N/A")
            atr  = indicators.get("ATR", "N/A")
            earnings_note = fetch_earnings_date(ticker)
            option_idea = fetch_best_option_from_yahoo(ticker)

            if or_high and or_low:
                target_long  = round(or_high * 1.02, 2)
                stop_long    = round(or_low, 2)
                target_short = round(or_low  * 0.98, 2)
                stop_short   = round(or_high, 2)
                insight_lines.append(
                    f"**{ticker}** → Long: 📈 Entry OR-high ${or_high} → Target: ${target_long}, Stop: ${stop_long} | RSI: {rsi} | MACD: {macd} | ATR: {atr} {earnings_note}"
                )
                insight_lines.append(
                    f"**{ticker}** → Short: 📉 Entry OR-low ${or_low} → Target: ${target_short}, Stop: ${stop_short} | RSI: {rsi} | MACD: {macd} | ATR: {atr} {earnings_note}"
                )
                insight_lines.append(option_idea)
            else:
                insight_lines.append(
                    f"**{ticker}** → OR levels not available. | RSI: {rsi} | MACD: {macd} | ATR: {atr} {earnings_note}"
                )
                insight_lines.append(option_idea)

    return "\n".join(insight_lines)

# ============================
# Scheduled summary generator
# ============================
def minutes_since_open(now_et: datetime) -> float:
    mo = now_et.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
    return (now_et - mo).total_seconds() / 60.0

def build_scheduled_summary_for_tickers(tickers):
    """Compute ORH/ORL from today's first 30m and bucket by live price."""
    summary = {"🚨 Extremely Weak": [], "⚠️ Weak": [], "🙀 Testing OR": [], "✅ Above OR": []}
    for tkr in tickers:
        orh, orl = compute_opening_range_30m(tkr)
        last = get_last_price(tkr)
        content = {"raw": [f"{tkr} (scheduled compute)"], "OR_high": orh, "OR_low": orl}
        if orh and orl and last is not None:
            if last >= orh:
                summary["✅ Above OR"].append((tkr, content))
            elif last <= orl:
                summary["🚨 Extremely Weak"].append((tkr, content))
            else:
                summary["🙀 Testing OR"].append((tkr, content))
        else:
            summary["🙀 Testing OR"].append((tkr, content))
    return summary

# ============================
# Discord client
# ============================
class MarketPulseClient(discord.Client):
    async def on_ready(self):
        info(f"[READY] Logged in as {self.user} (ID: {self.user.id})")

    async def on_message(self, message: discord.Message):
        if message.channel.id != READ_CHANNEL_ID:
            return

        txt = (message.content or "").strip()

        # ---- Manual diagnostic command: !diag TICKER ----
        if txt.lower().startswith("!diag"):
            parts = txt.split()
            ticker = parts[1].upper() if len(parts) > 1 else "SPY"
            info(f"[DIAG] Manual diagnostic for {ticker}")
            rsi = fetch_rsi(ticker)
            ind = fetch_additional_indicators(ticker)
            earn = fetch_earnings_date(ticker)
            opt = fetch_best_option_from_yahoo(ticker)
            out = (
                f"**Diagnostic – {ticker}**\n"
                f"RSI(14): {rsi} | ATR: {ind.get('ATR')} | MACD(hist): {ind.get('MACD')} {earn}\n"
                f"{opt}"
            )
            await message.channel.send(out)
            return

        # ---- Status-block trigger ----
        content = txt
        if not content and message.embeds:
            embed = message.embeds[0]
            content = embed.description.strip() if embed.description else ""

        if "status at" in (content or "").lower():
            info("[TRIGGER] Market pulse block detected. Processing…")
            summary   = classify_ticker_block(content)
            formatted = build_summary_message(summary)
            insight   = generate_insight(summary)
            full_message = formatted + "\n" + insight
            write_channel = self.get_channel(WRITE_CHANNEL_ID)
            if write_channel:
                await write_channel.send(full_message)
                info(f"✅ Posted summary to channel {WRITE_CHANNEL_ID}")
            else:
                warn("Write channel not found. Check POST_CHANNEL_ID.")

# Heartbeat task
async def heartbeat():
    while True:
        info("…still running (heartbeat). Waiting for new market pulse messages.")
        await asyncio.sleep(HEARTBEAT_SECS)

# Scheduler task: 30m for first 2.5h, then hourly, during market hours
async def scheduled_updates(client: discord.Client):
    while True:
        now_et = datetime.now(MARKET_TZ)
        today_open  = now_et.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
        today_close = now_et.replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0)

        if today_open.time() <= now_et.time() <= today_close.time():
            mins = minutes_since_open(now_et)
            interval = EARLY_INTERVAL if mins <= EARLY_MINUTES else LATER_INTERVAL
            info(f"[SCHED] Running scheduled market pulse. Next in {interval} minutes. ET now={now_et.strftime('%H:%M')}")

            try:
                tickers   = DEFAULT_TICKERS
                summary   = build_scheduled_summary_for_tickers(tickers)
                formatted = build_summary_message(summary)
                insight   = generate_insight(summary)
                payload   = f"{formatted}\n{insight}"
                write_channel = client.get_channel(WRITE_CHANNEL_ID)
                if write_channel:
                    await write_channel.send(payload)
                    info(f"✅ [SCHED] Posted scheduled summary to channel {WRITE_CHANNEL_ID}")
                else:
                    warn("[SCHED] Write channel not found. Check POST_CHANNEL_ID.")
            except Exception as e:
                error(f"[SCHED ERROR] {e}")

            await asyncio.sleep(interval * 60)
        else:
            # Sleep until the next market open
            next_open = today_open if now_et.time() < today_open.time() else (today_open + timedelta(days=1))
            sleep_secs = max(5, (next_open - now_et).total_seconds())
            info(f"[SCHED] Market closed. Sleeping until next open ({next_open.strftime('%Y-%m-%d %H:%M %Z')}) ~ {int(sleep_secs/60)} min.")
            await asyncio.sleep(sleep_secs)

client = MarketPulseClient(intents=intents)

async def main():
    info("Starting Discord client…")
    asyncio.create_task(heartbeat())
    asyncio.create_task(scheduled_updates(client))
    await client.start(BOT_TOKEN)

try:
    asyncio.run(main())
except KeyboardInterrupt:
    warn("Script interrupted by user. Exiting…")
except Exception as e:
    error(f"[FATAL] {e}")
