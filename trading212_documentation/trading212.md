# Trading 212 Public API Documentation (v0)

## Overview

Welcome to the official documentation for the Trading 212 Public API! This guide provides all the information you need to start building your own trading applications and integrations.

### General Information

This API is currently in **beta** and is under active development. We're continuously adding new features and improvements, and we welcome your feedback.

### API Environments

We provide two distinct environments for development and trading:

- **Paper Trading (Demo)**: `https://demo.trading212.com/api/v0`
- **Live Trading (Real Money)**: `https://live.trading212.com/api/v0`

You can test your applications extensively in the paper trading environment without risking real funds before moving to live trading.

> ⚠️ **Beta Limitations**: Please be aware that for the live (real money) environment, only Market Orders are supported for execution via the API at this time.

### Key Concepts

- **Authentication**: Every request to the API must be authenticated using a secure key pair
- **Rate Limiting**: All API calls are subject to rate limits to ensure fair usage and stability
- **IP Restrictions**: For enhanced security, you can optionally restrict your API keys to a specific set of IP addresses from within your Trading 212 account settings
- **Selling Orders**: To execute a sell order, you must provide a **negative value** for the quantity parameter (e.g., `-10.5`)

## Quickstart 🚀

This simple example shows you how to retrieve your account's cash balance.

First, generate your API keys from within the Trading 212 mobile app. For detailed instructions, visit: [How to get your Trading 212 API key](https://www.trading212.com/en/Trade-Equities/Trading-212-API)

Once you have your API Key and API Secret, you can make your first call:

```bash
# Step 1: Replace with your actual credentials and Base64-encode them
# The `-n` is important as it prevents adding a newline character
CREDENTIALS=$(echo -n "<YOUR_API_KEY>:<YOUR_API_SECRET>" | base64)

# Step 2: Make the API call to the live environment using the encoded credentials
curl -X GET "https://live.trading212.com/api/v0/equity/account/cash" \
     -H "Authorization: Basic $CREDENTIALS"
```

## Authentication 🔑

The API uses a secure key pair for authentication on every request. You must provide your API Key as the username and your API Secret as the password, formatted as an HTTP Basic Authentication header.

The Authorization header is constructed by Base64-encoding your `API_KEY:API_SECRET` string and prepending it with `Basic`.

### Building the Authorization Header

#### Linux or macOS (Terminal)

```bash
# This command outputs the required Base64-encoded string for your header
echo -n "<YOUR_API_KEY>:<YOUR_API_SECRET>" | base64
```

#### Python

```python
import base64

# 1. Your credentials
api_key = "<YOUR_API_KEY>"
api_secret = "<YOUR_API_SECRET>"

# 2. Combine them into a single string
credentials_string = f"{api_key}:{api_secret}"

# 3. Encode the string to bytes, then Base64 encode it
encoded_credentials = base64.b64encode(credentials_string.encode('utf-8')).decode('utf-8')

# 4. The final header value
auth_header = f"Basic {encoded_credentials}"
print(auth_header)
```

## Rate Limiting 🚦

To ensure high performance and fair access for all users, all API endpoints are subject to rate limiting.

**IMPORTANT NOTE**: All rate limits are applied on a per-account basis, regardless of which API key is used or which IP address the request originates from.

### Response Headers

Every API response includes the following headers to help you manage your request frequency:

- `x-ratelimit-limit`: The total number of requests allowed in the current time period
- `x-ratelimit-period`: The duration of the time period in seconds
- `x-ratelimit-remaining`: The number of requests you have left in the current period
- `x-ratelimit-reset`: A Unix timestamp indicating the exact time when the limit will be fully reset
- `x-ratelimit-used`: The number of requests you have already made in the current period

### How It Works

The rate limiter allows for requests to be made in bursts. For example, an endpoint with a limit of 50 requests per 1 minute does not strictly mean you can only make one request every 1.2 seconds. Instead, you could:

- Make a burst of all 50 requests in the first 5 seconds of a minute, then wait for the reset time
- Pace your requests evenly, for example, by making one call every 1.2 seconds

### Function-Specific Limits

In addition to the general rate limits on HTTP calls, some actions have their own functional limits. For example, there is a maximum of **50 pending orders** allowed per ticker, per account.

## Useful Links 🔗

- [Trading 212 API Terms](https://www.trading212.com/en/Terms-and-Conditions)
- [Trading 212 Community Forum](https://community.trading212.com/) - A great place to ask questions and share what you've built

## API Endpoints

### Pies

Manage your investment Pies. Use these endpoints to create, view, update, and delete your custom portfolios.

#### Fetch all pies

- **GET** `/api/v0/equity/pies`
- **Rate Limit**: 1 / 30s
- **Required Scope**: `pies:read`

#### Create pie

- **POST** `/api/v0/equity/pies`
- **Rate Limit**: 1 / 5s
- **Required Scope**: `pies:write`

#### Delete pie

- **DELETE** `/api/v0/equity/pies/{id}`
- **Rate Limit**: 1 / 5s
- **Required Scope**: `pies:write`

#### Fetch a pie

- **GET** `/api/v0/equity/pies/{id}`
- **Rate Limit**: 1 / 5s
- **Required Scope**: `pies:read`

#### Update pie

- **POST** `/api/v0/equity/pies/{id}`
- **Rate Limit**: 1 / 5s
- **Required Scope**: `pies:write`

#### Duplicate pie

- **POST** `/api/v0/equity/pies/{id}/duplicate`
- **Rate Limit**: 1 / 5s
- **Required Scope**: `pies:write`

### Equity Orders

Place, monitor, and cancel equity trade orders. This section provides the core functionality for programmatically executing your trading strategies for stocks and ETFs.

#### Get All Pending Orders

- **GET** `/api/v0/equity/orders`
- **Rate Limit**: 1 / 5s
- **Required Scope**: `orders:read`

Retrieves a list of all orders that are currently active (not yet filled, cancelled, or expired).

#### Place a Limit Order

- **POST** `/api/v0/equity/orders/limit`
- **Rate Limit**: 1 / 2s
- **Required Scope**: `orders:execute`

Creates a new Limit order, which executes at a specified price or better.

- Use positive quantity for BUY orders
- Use negative quantity for SELL orders

#### Place a Market Order

- **POST** `/api/v0/equity/orders/market`
- **Rate Limit**: 50 / 1m
- **Required Scope**: `orders:execute`

Creates a new Market order for immediate execution at the next available price.

#### Place a Stop Order

- **POST** `/api/v0/equity/orders/stop`
- **Rate Limit**: 1 / 2s
- **Required Scope**: `orders:execute`

Creates a new Stop order, which places a Market order once the stopPrice is reached.

#### Place a Stop-Limit Order

- **POST** `/api/v0/equity/orders/stop_limit`
- **Rate Limit**: 1 / 2s
- **Required Scope**: `orders:execute`

Creates a new Stop-Limit order, combining features of Stop and Limit orders.

#### Cancel a Pending Order

- **DELETE** `/api/v0/equity/orders/{id}`
- **Rate Limit**: 50 / 1m
- **Required Scope**: `orders:execute`

Attempts to cancel an active, unfilled order by its unique ID.

#### Get Order by ID

- **GET** `/api/v0/equity/orders/{id}`
- **Rate Limit**: 1 / 1s
- **Required Scope**: `orders:read`

### Account Data

Access fundamental information about your trading account.

#### Get Account Cash Balance

- **GET** `/api/v0/equity/account/cash`
- **Rate Limit**: 1 / 2s
- **Required Scope**: `account`

#### Get Account Information

- **GET** `/api/v0/equity/account/info`
- **Rate Limit**: 1 / 30s
- **Required Scope**: `account`

### Personal Portfolio

View the current state of your portfolio.

#### Fetch all open positions

- **GET** `/api/v0/equity/portfolio`
- **Rate Limit**: 1 / 5s
- **Required Scope**: `portfolio`

#### Search for a specific position by ticker

- **POST** `/api/v0/equity/portfolio/ticker`
- **Rate Limit**: 1 / 1s
- **Required Scope**: `portfolio`

#### Fetch a specific position

- **GET** `/api/v0/equity/portfolio/{ticker}`
- **Rate Limit**: 1 / 1s
- **Required Scope**: `portfolio`

### Instruments Metadata

Discover what you can trade.

#### Exchange List

- **GET** `/api/v0/equity/metadata/exchanges`
- **Rate Limit**: 1 / 30s
- **Required Scope**: `metadata`

#### Instrument List

- **GET** `/api/v0/equity/metadata/instruments`
- **Rate Limit**: 1 / 50s
- **Required Scope**: `metadata`

### Historical Items

Review your account's trading history.

#### Historical order data

- **GET** `/api/v0/equity/history/orders`
- **Rate Limit**: 6 / 1m
- **Required Scope**: `history:orders`

#### Paid out dividends

- **GET** `/api/v0/history/dividends`
- **Rate Limit**: 6 / 1m
- **Required Scope**: `history:dividends`

#### List Generated Reports

- **GET** `/api/v0/history/exports`
- **Rate Limit**: 1 / 1m

#### Request a CSV Report

- **POST** `/api/v0/history/exports`
- **Rate Limit**: 1 / 30s

Initiates the generation of a CSV report containing historical account data. This is an asynchronous operation.

#### Transaction list

- **GET** `/api/v0/history/transactions`
- **Rate Limit**: 6 / 1m
- **Required Scope**: `history:transactions`

---

For more details and examples, refer to the full API specification or visit the [Trading 212 Community Forum](https://community.trading212.com/).
