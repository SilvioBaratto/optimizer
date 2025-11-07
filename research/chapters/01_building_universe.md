# Building the Universe

The construction of a robust investment universe forms the foundation of any quantitative portfolio optimization system. This chapter describes the methodology employed to build a comprehensive, high-quality dataset of financial instruments suitable for institutional-grade portfolio analysis. The process integrates multiple data sources, applies rigorous filtering criteria, and implements efficient data enrichment techniques to ensure the resulting universe meets institutional standards for liquidity, data completeness, and historical coverage.

## Conceptual Framework

The universe construction process can be conceptualized as a multi-stage filtering pipeline that progressively refines a broad set of available securities into a focused collection of investment-grade instruments. The methodology draws inspiration from institutional investment practices, where asset selection begins with establishing minimum standards for market capitalization, liquidity, and data availability before proceeding to more sophisticated quantitative analysis.

The universe building methodology addresses three fundamental challenges:

1. **Data Integration**: Harmonizing instrument identifiers across disparate data providers
2. **Quality Assurance**: Establishing and enforcing minimum standards for inclusion
3. **Scalability**: Processing thousands of instruments efficiently while maintaining data integrity

## Data Sources and Geographic Scope

### Primary Data Providers

The system integrates two complementary data sources to achieve comprehensive coverage:

**Trading212 API** serves as the primary source for instrument metadata, providing access to thousands of publicly traded equities across major global exchanges. The API delivers structured information including ticker symbols, ISIN codes, exchange affiliations, and trading availability status. This source ensures the universe reflects instruments actually available for trading through modern brokerage platforms.

**Yahoo Finance (yfinance)** functions as the enrichment layer, augmenting basic instrument metadata with comprehensive fundamental, technical, and financial data. This includes historical price series, financial statement data, valuation ratios, profitability metrics, and analyst coverage. The depth of yfinance data enables rigorous filtering and subsequent quantitative analysis.

### Geographic Coverage

The investment universe targets developed and emerging markets aligned with institutional geographic allocation frameworks. The selected countries reflect both economic significance and data availability:

- **United States**: 55-65% target allocation, representing the deepest and most liquid equity markets with superior data quality
- **Europe**: 15-20% aggregate allocation across Germany, France, and United Kingdom
- **Japan**: 8-12% allocation, capturing Asia-Pacific developed market exposure
- **Emerging Markets**: 6-8% allocation to China and India where data availability permits

This geographic distribution aligns with standard institutional benchmarks while acknowledging data provider constraints that limit comprehensive coverage of certain emerging markets.

## Ticker Mapping Methodology

### The Identifier Reconciliation Problem

A fundamental challenge in multi-source financial data integration involves reconciling divergent ticker symbol conventions across data providers. Trading212 employs exchange-specific ticker formats, while Yahoo Finance utilizes a distinct suffix notation system to denote exchange affiliation. For example, a stock listed on the London Stock Exchange might appear as "VOD" in Trading212 but requires the suffix ".L" (VOD.L) for yfinance queries.

### Systematic Ticker Discovery

The ticker mapping process implements a systematic discovery algorithm that attempts to identify the correct yfinance ticker for each Trading212 instrument. The procedure follows a prioritized search strategy:

1. **Cache Lookup**: Check persistent mapping cache for previously validated ticker pairs
2. **Exchange-Specific Mapping**: Apply known suffix rules based on exchange affiliation (e.g., ".L" for London Stock Exchange, ".PA" for Euronext Paris, no suffix for US exchanges)
3. **Symbol Normalization**: Transform ticker format conventions (e.g., slash-to-dash conversion for share classes)
4. **Validation**: Verify mapped ticker returns valid market data from yfinance

The mapping cache reduces redundant API calls and improves processing efficiency. Cached mappings are time-limited (90-day expiration) to ensure stale mappings do not persist indefinitely.

### Rate Limiting and Error Handling

To maintain system stability and comply with API usage policies, the ticker discovery process implements adaptive rate limiting and exponential backoff strategies. The rate limiter enforces a maximum query frequency of 5 requests per second, while the backoff mechanism progressively increases wait times when encountering rate limit errors:

$$
t_{\text{wait}} = \min(2^n \cdot t_{\text{base}}, t_{\text{max}})
$$

where $n$ represents the retry attempt number, $t_{\text{base}}$ is the base wait interval, and $t_{\text{max}}$ caps the maximum wait time at one hour. This approach balances throughput with reliability.

## Institutional Filtering Framework

### Market Capitalization Tiers

The universe employs a tiered market capitalization classification that segments stocks into three categories, each with distinct liquidity requirements:

$$
\text{Segment}(i) = \begin{cases}
\text{Large-cap} & \text{if } \text{MarketCap}_i \geq \$10\text{B} \\
\text{Mid-cap} & \text{if } \$2\text{B} \leq \text{MarketCap}_i < \$10\text{B} \\
\text{Small-cap} & \text{if } \$100\text{M} \leq \text{MarketCap}_i < \$2\text{B}
\end{cases}
$$

The absolute minimum market capitalization threshold of \$100 million excludes micro-cap stocks that typically exhibit insufficient liquidity and elevated transaction costs for institutional portfolios.

### Liquidity Requirements

Liquidity filtering applies segment-specific thresholds that recognize the relationship between firm size and trading volume. The daily dollar volume requirement for stock $i$ is defined as:

$$
\text{DollarVolume}_i = \text{AverageVolume}_i \times \text{Price}_i
$$

where $\text{AverageVolume}_i$ represents the mean daily share volume over the preceding period. The minimum thresholds vary by market capitalization segment:

| Segment | Minimum Dollar Volume | Minimum Share Volume |
|---------|----------------------|---------------------|
| Large-cap | \$10,000,000 | 500,000 shares |
| Mid-cap | \$5,000,000 | 250,000 shares |
| Small-cap | \$1,000,000 | 100,000 shares |

This tiered structure ensures that portfolio construction algorithms can execute realistic position sizes without excessive market impact, while avoiding overly restrictive criteria that would eliminate otherwise suitable smaller-capitalization opportunities.

### Price Filters

To exclude problematic securities, the system enforces price bounds:

$$
\$5 \leq \text{Price}_i \leq \$10,000
$$

The lower bound eliminates penny stocks that exhibit elevated bid-ask spreads and susceptibility to manipulation. The upper bound serves as a data quality check, flagging potential data errors or corporate actions not yet reflected in databases.

### Data Completeness Requirements

Institutional quantitative analysis requires comprehensive fundamental and technical data across multiple dimensions. The system defines eleven data categories, each comprising specific required fields:

1. **Market Capitalization**: Total market value
2. **Current Price**: Latest trading price
3. **Volume Metrics**: Average daily trading volume
4. **Share Structure**: Outstanding shares count
5. **Risk Metrics**: Market beta
6. **Classification**: GICS sector and industry
7. **Exchange Information**: Primary listing venue
8. **Valuation Ratios**: P/E ratio, Price-to-Book ratio
9. **Profitability**: Return on Equity, operating margins
10. **Balance Sheet**: Debt ratios, total assets
11. **Price Range**: 52-week high and low

A stock passes the data completeness filter only if 100% of required categories contain valid, non-null data. This stringent requirement ensures downstream analysis does not encounter missing data issues that could compromise signal quality or introduce bias.

### Historical Data Coverage

To support robust statistical estimation and backtesting, the system requires minimum historical price coverage. The historical data criterion is defined as:

$$
N_{\text{trading days}} \geq 750 \text{ days}
$$

This threshold corresponds to approximately three years of trading history (assuming 252 trading days per year), providing sufficient observations for:

- Long-term momentum pattern identification
- Risk parameter estimation (beta, volatility)
- Fundamental trend analysis
- Statistical significance testing

The implementation requests five years of historical data from yfinance (period='5y') and validates that at least 750 days are returned, accommodating potential gaps from market holidays, trading halts, or data provider limitations.

## Pipeline Architecture and Processing Strategy

### Sequential Exchange Processing with Concurrent Enrichment

The universe construction pipeline employs a hybrid processing strategy that balances computational efficiency with system stability. Processing proceeds sequentially across exchanges but leverages concurrent workers within each exchange to parallelize instrument enrichment.

For each exchange $E$, let $I_E = \{i_1, i_2, \ldots, i_n\}$ represent the set of instruments listed on that exchange. The enrichment process distributes instruments across $k$ concurrent worker threads:

$$
\text{Throughput} = \frac{k}{t_{\text{avg}}}
$$

where $t_{\text{avg}}$ represents the average processing time per instrument, including API latency and data validation. Empirical testing established $k=20$ concurrent workers as optimal, achieving 20-50 instruments per second throughput while maintaining acceptable error rates.

### Short-Lived Database Sessions

To mitigate SSL timeout issues inherent in long-running database connections, the system employs short-lived database sessions scoped to specific operations:

1. **Exchange Creation**: New session per exchange for metadata insertion
2. **Batch Insert**: New session per batch (50-500 instruments) for bulk saves
3. **Reporting**: New session for final statistics generation

This session management strategy trades slight overhead for improved reliability, particularly important when processing thousands of instruments over extended periods.

### Batch Insertion for Performance

Rather than committing each instrument individually, the pipeline accumulates processed instruments into batches before database persistence:

$$
N_{\text{batches}} = \left\lceil \frac{N_{\text{instruments}}}{B} \right\rceil
$$

where $B$ represents the batch size (typically 50-500 instruments). Batching reduces transaction overhead and database round-trips, significantly improving overall throughput.

## Quality Assurance and Validation

### Multi-Stage Validation

Each instrument undergoes validation at multiple pipeline stages:

1. **Ticker Discovery Validation**: Verify yfinance returns valid market data for mapped ticker
2. **Data Fetch Validation**: Confirm fundamental data response contains minimum field count
3. **Filter Validation**: Apply all institutional filters with explicit pass/fail determination
4. **Historical Data Validation**: Confirm sufficient trading history availability

Only instruments passing all validation stages enter the final universe. Failed instruments are logged with specific rejection reasons to support pipeline debugging and filter refinement.

### Data Quality Metrics

The system tracks several quality metrics throughout the processing pipeline:

**Mapping Success Rate**: The proportion of Trading212 instruments successfully mapped to yfinance tickers, typically achieving 85-95% depending on exchange and asset type.

**Filter Pass Rate**: The proportion of successfully mapped instruments that satisfy all institutional filters. This rate varies significantly by exchange, with major US exchanges (NYSE, NASDAQ) exhibiting higher pass rates (40-60%) than smaller international venues (10-30%).

**Data Completeness Distribution**: Histogram of data coverage across instruments, identifying systematic data gaps by exchange or sector.

## Performance Characteristics and Scalability

### Processing Throughput

Under typical operating conditions with 20 concurrent workers and stable API performance, the system achieves:

- **Ticker Mapping**: 30-50 instruments/second
- **Data Enrichment**: 20-40 instruments/second (including filter validation)
- **Total Processing Time**: 10-15 minutes for 1,000 instruments

These rates fluctuate based on network latency, API response times, and data provider rate limiting.

### Cache Efficiency

The persistent ticker mapping cache significantly improves performance on subsequent runs. Cache hit rates typically reach 60-70% after initial universe construction, reducing redundant API calls and accelerating incremental updates.

### Resource Utilization

The concurrent architecture achieves high CPU utilization during processing phases, while I/O wait time dominates overall execution due to network API calls. Memory usage remains modest (< 2GB) even for large universes, as the pipeline processes instruments in streaming fashion rather than loading entire datasets into memory.

## Challenges and Limitations

### Ticker Mapping Ambiguity

Certain instruments present mapping challenges:

- **Multiple Share Classes**: Companies with multiple share classes (e.g., Class A vs. Class B) require careful suffix handling
- **ADR vs. Ordinary Shares**: American Depositary Receipts may map to either ADR tickers or underlying ordinary shares
- **Recent Corporate Actions**: Ticker changes, mergers, and spin-offs may cause temporary mapping failures

The system handles these cases through manual override mappings and periodic cache validation.

### Data Provider Inconsistencies

Discrepancies occasionally arise between Trading212 and yfinance data:

- **Stale Fundamental Data**: yfinance may report outdated fundamentals for thinly traded stocks
- **Exchange Mismatches**: Primary listing venue may differ between providers
- **Currency Reporting**: Inconsistent currency normalization across providers

These issues are addressed through data prioritization hierarchies and conflict logging for manual review.

### API Rate Limiting

Both Trading212 and yfinance impose rate limits that constrain processing throughput. The exponential backoff strategy mitigates rate limit errors but can extend total processing time significantly when limits are repeatedly encountered.

### Geographic Coverage Gaps

The reliance on Trading212 as the primary instrument source limits coverage to exchanges supported by that platform. Notably absent:

- **Direct China A-Shares**: Only available through ADRs or Hong Kong listings
- **India Local Markets**: Limited to large-cap names with international listings
- **Smaller Emerging Markets**: Minimal coverage of Southeast Asia, Latin America, Africa

Future enhancements could integrate additional data sources to expand geographic reach.

## Summary

The universe construction methodology successfully builds a high-quality dataset of investment-grade equities suitable for institutional portfolio optimization. The multi-stage filtering framework ensures all instruments meet minimum standards for market capitalization, liquidity, data completeness, and historical coverage. The parallel processing architecture achieves acceptable throughput while maintaining system stability through rate limiting and short-lived database sessions.

The resulting universe typically contains 500-1,500 instruments (depending on market conditions and filter strictness), representing liquid, well-documented securities across developed and select emerging markets. This filtered universe forms the foundation for subsequent macro regime analysis and stock signal generation, as described in the following chapters.
