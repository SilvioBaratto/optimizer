# WebSocket — real-time quotes

Two classes: `WebSocket` (synchronous) and `AsyncWebSocket` (async). Both stream JSON price
updates from Yahoo. Convenience entry points `Ticker.live()` and `Tickers.live()` wrap them.

> **API note (1.6.0):** there is **no `run()` method and no `on_message` attribute**. You call
> `subscribe(...)`, then `listen(message_handler)` (the handler is an *argument* to `listen`,
> optional), and `close()`. Both classes support context-manager usage.

## Constructor

```python
yf.WebSocket(url="wss://streamer.finance.yahoo.com/?version=2", verbose=True)
yf.AsyncWebSocket(url="wss://streamer.finance.yahoo.com/?version=2", verbose=True)
```

| Param | Default | Meaning |
|---|---|---|
| `url` | `wss://streamer.finance.yahoo.com/?version=2` | Yahoo streamer endpoint (protocol `version=2`) |
| `verbose` | `True` | Print status messages; set `False` to silence |

## Methods

| Method | Sync | Async | Meaning |
|---|---|---|---|
| `subscribe(symbols)` | `subscribe([...])` | `await subscribe([...])` | Subscribe to a symbol or list |
| `unsubscribe(symbols)` | `unsubscribe([...])` | `await unsubscribe([...])` | Remove symbols |
| `listen(message_handler=None)` | `listen(handler)` | `await listen(handler)` | Start receiving; invokes `handler(msg)` per message. Blocks/loops |
| `close()` | `close()` | `await close()` | Close the connection |

`symbols` is `str | list[str]`. `message_handler` is `Callable[[dict], None] | None`.

## Synchronous

```python
import yfinance as yf

def on_message(msg):
    # msg keys (snake_case proto names): id (symbol), price, time, day_volume,
    # change, change_percent, market_hours, quote_type, exchange, ...
    print(f"{msg['id']}: {msg['price']}  vol={msg.get('day_volume')}")

# Context manager handles close() automatically
with yf.WebSocket() as ws:
    ws.subscribe(["AAPL", "MSFT"])
    ws.listen(on_message)          # blocks — pass the handler here, not via on_message=
```

Without a context manager, call `ws.close()` yourself when done.

## Asynchronous

```python
import asyncio
import yfinance as yf

async def on_message(msg):
    print(msg)

async def main():
    async with yf.AsyncWebSocket() as ws:
        await ws.subscribe(["AAPL", "MSFT"])
        await ws.listen(on_message)     # awaited; handler optional

asyncio.run(main())
```

> **Jupyter:** nested event loops may need `nest_asyncio`. The async client's
> `subscribe` / `unsubscribe` / `listen` / `close` are coroutines — always `await` them.

## Ticker.live() convenience

```python
t = yf.Ticker("AAPL")
t.live()          # streams this ticker

ts = yf.Tickers("AAPL MSFT GOOG")
ts.live()         # batch streaming across the container
```

`live()` is surfaced on both `Ticker` and `Tickers` as a first-class streaming shortcut over
the WebSocket clients.

## Message shape

```json
{
  "id": "AAPL",
  "price": 203.45,
  "time": "1702834512345",
  "day_volume": "39820145",
  "change": 1.23,
  "change_percent": 0.61,
  "market_hours": 1,
  "exchange": "NMS",
  "quote_type": 8,
  "currency": "USD"
}
```

Not every field appears in every tick — treat missing fields as "no update to that attribute".

## When to prefer WebSocket vs. polling

- **Poll** (`fast_info.last_price` in a loop) for a handful of tickers at minute-or-slower cadence — simpler, no connection management.
- **WebSocket** when you need sub-second updates or are streaming > ~20 symbols — avoids rate limits and cuts latency.
