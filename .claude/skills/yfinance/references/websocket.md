# WebSocket — real-time quotes

Two classes: `WebSocket` (blocking) and `AsyncWebSocket` (async iterator). Both stream the same JSON price updates from Yahoo.

## Synchronous

```python
import yfinance as yf

def on_message(ws, msg):
    # msg keys: id (symbol), price, time, dayVolume, change, changePercent, ...
    print(f"{msg['id']}: {msg['price']}  vol={msg['dayVolume']}")

ws = yf.WebSocket()
ws.on_message = on_message          # bind handler before subscribe / run
ws.subscribe(["AAPL", "MSFT"])
ws.run()                            # blocks
```

Bind the handler **before** calling `subscribe` / `run` — if you set it after, the first few messages that arrive before the assignment will be dropped silently.

## Asynchronous

```python
import asyncio
import yfinance as yf

async def main():
    ws = yf.AsyncWebSocket()
    ws.subscribe(["AAPL", "MSFT"])

    async for msg in ws:
        print(msg)

asyncio.run(main())
```

The async form is usually easier to stop: break out of the loop or cancel the task.

## Message shape

```json
{
  "id": "AAPL",
  "price": 203.45,
  "time": "1702834512345",
  "dayVolume": "39820145",
  "change": 1.23,
  "changePercent": 0.61,
  "marketHours": "REGULAR_MARKET",
  "exchange": "NMS",
  "quoteType": "EQUITY",
  "currency": "USD"
}
```

Not every field appears in every tick — treat missing fields as "no update to that attribute".

## Subscribing & unsubscribing dynamically

```python
ws.subscribe(["GOOG"])              # add
ws.unsubscribe(["AAPL"])            # remove
```

Works on both sync and async variants.

## When to prefer WebSocket vs. polling

- **Poll** (`fast_info.lastPrice` in a loop) for a handful of tickers at minute-or-slower cadence — simpler, no connection management.
- **WebSocket** when you need sub-second updates or are streaming > ~20 symbols — avoids rate limits and cuts latency.
