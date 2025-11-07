// ============================================================================
// COMPREHENSIVE YFINANCE STOCK DATA TYPES
// All field names match exactly with yfinance API
// Based on yfinance_metadata_report.txt analysis
// ============================================================================

// ============================================================================
// 1. CORE PRICE & VALUATION DATA (34 fields from yfinance)
// ============================================================================

class YFinancePriceData {
  // Current Price Data
  currentPrice float?
  regularMarketPrice float?
  regularMarketOpen float?
  open float?
  previousClose float?
  regularMarketPreviousClose float?

  // Intraday Range
  dayHigh float?
  dayLow float?
  regularMarketDayHigh float?
  regularMarketDayLow float?
  regularMarketDayRange string?

  // 52-Week Range
  fiftyTwoWeekHigh float?
  fiftyTwoWeekLow float?
  fiftyTwoWeekHighChange float?
  fiftyTwoWeekLowChange float?
  fiftyTwoWeekHighChangePercent float?
  fiftyTwoWeekLowChangePercent float?
  fiftyTwoWeekRange string?
  fiftyTwoWeekChangePercent float?  // 52WeekChange from yfinance

  // All-Time Highs/Lows
  allTimeHigh float?
  allTimeLow float?

  // Bid/Ask
  bid float?
  ask float?
  bidSize int?
  askSize int?

  // Market Changes
  regularMarketChange float?
  regularMarketChangePercent float?

  // Post-Market (if available)
  postMarketPrice float?
  postMarketChange float?
  postMarketChangePercent float?
  postMarketTime int?

  // Analyst Targets
  targetMeanPrice float?
  targetMedianPrice float?
  targetHighPrice float?
  targetLowPrice float?

  // Valuation Ratios
  priceToBook float?
  priceToSalesTrailing12Months float?
  priceEpsCurrentYear float?

  // Other
  priceHint int?
  customPriceAlertConfidence string?
}

// ============================================================================
// 2. FINANCIAL METRICS (28 fields from yfinance)
// ============================================================================

class YFinanceFinancialMetrics {
  // Profitability Ratios
  profitMargins float?
  grossMargins float?
  operatingMargins float?
  ebitdaMargins float?

  // Efficiency Metrics
  returnOnAssets float?
  returnOnEquity float?

  // Growth Metrics
  revenueGrowth float?
  earningsGrowth float?
  earningsQuarterlyGrowth float?

  // Liquidity Ratios
  currentRatio float?
  quickRatio float?

  // Leverage
  debtToEquity float?

  // Dividend Metrics
  payoutRatio float?

  // Absolute Financials
  grossProfits float?
  totalRevenue float?
  revenuePerShare float?
  netIncomeToCommon float?
  totalCash float?
  totalCashPerShare float?
  totalDebt float?

  // Cash Flow
  operatingCashflow float?
  freeCashflow float?

  // Valuation
  enterpriseToRevenue float?

  // Short Interest
  shortRatio float?

  // Earnings Timing
  earningsTimestampStart int?
  earningsTimestampEnd int?
  earningsTimestamp int?
  earningsCallTimestampStart int?
  earningsCallTimestampEnd int?
  isEarningsDateEstimate bool?

  // Other
  trailingPegRatio float?
}

// ============================================================================
// 3. DIVIDEND & YIELD DATA (9 fields from yfinance)
// ============================================================================

class YFinanceDividendData {
  dividendYield float?
  dividendRate float?
  trailingAnnualDividendRate float?
  trailingAnnualDividendYield float?
  fiveYearAvgDividendYield float?

  // Dividend Dates
  dividendDate int?  // Unix timestamp
  exDividendDate int?
  lastDividendDate int?
  lastDividendValue float?
}

// ============================================================================
// 4. COMPANY INFORMATION (20 fields from yfinance)
// ============================================================================

class YFinanceCompanyInfo {
  // Identifiers
  symbol string?
  shortName string?
  longName string?
  displayName string?

  // Classification
  sector string?
  sectorDisp string?
  sectorKey string?
  industry string?
  industryDisp string?
  industryKey string?

  // Location
  country string?
  state string?
  city string?
  zip string?
  address1 string?
  address2 string?

  // Contact
  phone string?
  website string?
  irWebsite string?

  // Description
  longBusinessSummary string?

  // Exchange Info
  exchange string?
  fullExchangeName string?
  quoteSourceName string?
  exchangeTimezoneName string?
  exchangeTimezoneShortName string?

  // Employees
  fullTimeEmployees int?
}

// ============================================================================
// 5. OFFICERS & GOVERNANCE (6 fields from yfinance)
// ============================================================================

class YFinanceGovernance {
  // Officers (list of dicts with name, title, age, etc)
  companyOfficers string?  // JSON string of officer list
  executiveTeam string?    // JSON string of executive data

  // Governance Scores
  boardRisk int?
  compensationRisk int?
  overallRisk int?
  shareHolderRightsRisk int?
  auditRisk int?

  // Compensation Date
  compensationAsOfEpochDate int?
  governanceEpochDate int?

  // Communication
  messageBoardId string?
}

// ============================================================================
// 6. MARKET DATA & VOLUME (25 fields from yfinance)
// ============================================================================

class YFinanceMarketData {
  // Market Cap & Shares
  marketCap float?
  sharesOutstanding float?
  impliedSharesOutstanding float?
  floatShares float?

  // Volume
  volume float?
  regularMarketVolume float?
  averageVolume float?
  averageVolume10days float?
  averageDailyVolume10Day float?
  averageDailyVolume3Month float?

  // Short Interest
  sharesShort float?
  sharesShortPriorMonth float?
  shortPercentOfFloat float?
  sharesPercentSharesOut float?
  sharesShortPreviousMonthDate int?
  dateShortInterest int?

  // Institutional Holdings
  heldPercentInstitutions float?
  heldPercentInsiders float?

  // Market Info
  market string?
  marketState string?
  hasPrePostMarketData bool?

  // Timing
  regularMarketTime int?
  gmtOffSetMilliseconds int?

  // Other
  corporateActions string?  // JSON string
}

// ============================================================================
// 7. TRADING INFO (6 fields from yfinance)
// ============================================================================

class YFinanceTradingInfo {
  currency string?
  financialCurrency string?
  exchange string?
  exchangeDataDelayedBy int?
  tradeable bool?
  cryptoTradeable bool?
}

// ============================================================================
// 8. ANALYST RECOMMENDATIONS (4 fields from yfinance)
// ============================================================================

class YFinanceAnalystData {
  recommendationKey string?  // "buy", "strong_buy", "hold", "sell", "strong_sell"
  recommendationMean float?  // 1.0 (Strong Buy) to 5.0 (Strong Sell)
  numberOfAnalystOpinions int?
  averageAnalystRating string?
}

// ============================================================================
// 9. DATES & TIME (7 fields from yfinance)
// ============================================================================

class YFinanceDates {
  firstTradeDateMilliseconds int?
  lastFiscalYearEnd int?
  nextFiscalYearEnd int?
  mostRecentQuarter int?
  lastSplitDate int?
  lastSplitFactor string?
  ipoExpectedDate int?
}

// ============================================================================
// 10. TECHNICAL & OTHER (46 fields from yfinance)
// ============================================================================

class YFinanceTechnicalData {
  // Moving Averages
  fiftyDayAverage float?
  twoHundredDayAverage float?
  fiftyDayAverageChange float?
  twoHundredDayAverageChange float?
  fiftyDayAverageChangePercent float?
  twoHundredDayAverageChangePercent float?

  // Beta
  beta float?

  // Performance vs S&P 500
  SandP52WeekChange float?  // Exact field name with capital S and P

  // EPS Data
  epsTrailingTwelveMonths float?
  epsCurrentYear float?
  epsForward float?
  forwardEps float?
  trailingEps float?

  // P/E Ratios
  trailingPE float?
  forwardPE float?

  // Book Value
  bookValue float?

  // EBITDA & Enterprise Value
  ebitda float?
  enterpriseValue float?
  enterpriseToEbitda float?

  // ESG
  esgPopulated bool?

  // Quote Metadata
  quoteType string?
  typeDisp string?
  language string?
  region string?
  maxAge int?
  sourceInterval int?
  triggerable bool?

  // Other
  fax string?
}

// ============================================================================
// COMPLETE YFINANCE STOCK DATA (ALL CATEGORIES COMBINED)
// ============================================================================

class ComprehensiveYFinanceData {
  // Identifiers (required)
  ticker string
  company_name string

  // All yfinance categories (all optional since availability varies)
  price_data YFinancePriceData?
  financial_metrics YFinanceFinancialMetrics?
  dividend_data YFinanceDividendData?
  company_info YFinanceCompanyInfo?
  governance YFinanceGovernance?
  market_data YFinanceMarketData?
  trading_info YFinanceTradingInfo?
  analyst_data YFinanceAnalystData?
  dates YFinanceDates?
  technical_data YFinanceTechnicalData?

  // Metadata
  data_fetch_timestamp string
  data_completeness_pct float  // Percentage of fields populated
  yfinance_info_dict_size int  // Number of fields in raw yfinance info dict
}

// ============================================================================
// MINIMAL DATA FOR QUICK SCREENING
// ============================================================================

class MinimalYFinanceData {
  // Core identifiers
  ticker string
  company_name string
  sector string?
  industry string?

  // Essential price/valuation
  currentPrice float?
  marketCap float?
  volume float?

  // Key ratios (most stocks have these)
  trailingPE float?
  priceToBook float?
  returnOnEquity float?
  debtToEquity float?
  profitMargins float?

  // Growth
  revenueGrowth float?
  earningsGrowth float?

  // Performance
  fiftyTwoWeekChangePercent float?
  beta float?

  // Analyst consensus
  recommendationKey string?
}

// ============================================================================
// HISTORICAL PRICE DATA (OHLCV)
// ============================================================================

class HistoricalPriceData {
  // Time series data (parallel arrays)
  dates string[]  // ISO date strings
  open float[]
  high float[]
  low float[]
  close float[]
  adj_close float[]  // Adjusted for splits/dividends
  volume float[]

  // Metadata
  ticker string
  start_date string
  end_date string
  trading_days_count int
  data_quality_score float  // 0-1, based on completeness
}

// ============================================================================
// FINANCIAL STATEMENTS (from yfinance financials API)
// ============================================================================

class YFinanceIncomeStatement {
  periods string[]  // Quarter labels e.g., ["2024-09-30", "2024-06-30"]

  // Revenue
  TotalRevenue float[]?
  CostOfRevenue float[]?
  GrossProfit float[]?

  // Operating
  OperatingExpense float[]?
  OperatingIncome float[]?
  EBITDA float[]?

  // Net Income
  NetIncome float[]?
  NetIncomeCommonStockholders float[]?

  // Per Share
  BasicEPS float[]?
  DilutedEPS float[]?

  // Other
  ResearchAndDevelopment float[]?
  SellingGeneralAndAdministration float[]?
  InterestExpense float[]?
  TaxProvision float[]?
}

class YFinanceBalanceSheet {
  periods string[]

  // Assets
  TotalAssets float[]?
  CurrentAssets float[]?
  CashAndCashEquivalents float[]?
  CashCashEquivalentsAndShortTermInvestments float[]?
  AccountsReceivable float[]?
  Inventory float[]?
  PropertyPlantEquipment float[]?
  Goodwill float[]?
  IntangibleAssets float[]?

  // Liabilities
  TotalLiabilities float[]?
  CurrentLiabilities float[]?
  AccountsPayable float[]?
  ShortTermDebt float[]?
  LongTermDebt float[]?
  TotalDebt float[]?

  // Equity
  StockholdersEquity float[]?
  RetainedEarnings float[]?
  TreasuryStock float[]?

  // Shares
  ShareIssued float[]?
  OrdinarySharesNumber float[]?
}

class YFinanceCashFlow {
  periods string[]

  // Operating Activities
  OperatingCashFlow float[]?
  CashFlowFromContinuingOperatingActivities float[]?
  DepreciationAndAmortization float[]?
  ChangeInWorkingCapital float[]?

  // Investing Activities
  CashFlowFromInvestingActivities float[]?
  CapitalExpenditure float[]?
  NetBusinessPurchaseAndSale float[]?
  NetInvestmentPurchaseAndSale float[]?

  // Financing Activities
  CashFlowFromFinancingActivities float[]?
  RepurchaseOfCapitalStock float[]?
  RepaymentOfDebt float[]?
  IssuanceOfDebt float[]?
  CashDividendsPaid float[]?

  // Free Cash Flow (derived)
  FreeCashFlow float[]?
}

// ============================================================================
// COMPLETE FINANCIAL STATEMENTS BUNDLE
// ============================================================================

class YFinanceFinancialStatements {
  ticker string
  income_statement YFinanceIncomeStatement?
  balance_sheet YFinanceBalanceSheet?
  cash_flow YFinanceCashFlow?

  // Metadata
  statement_frequency string  // "quarterly" or "annual"
  most_recent_period string
  periods_available int
}
