export type ReportTemplate = 'executive_summary' | 'full_report' | 'risk_report' | 'performance_attribution';
export type ReportFormat = 'pdf' | 'xlsx' | 'csv';
export type ReportOrientation = 'portrait' | 'landscape';
export type ReportJobStatus = 'pending' | 'generating' | 'complete' | 'error';

export interface ReportSection {
  id: string;
  label: string;
  description: string;
  default: boolean;
}

export interface ReportBranding {
  companyName: string;
  primaryColor: string;
  includeDisclaimer: boolean;
}

export interface ReportConfig {
  template: ReportTemplate;
  sections: string[];
  branding: ReportBranding;
  format: ReportFormat;
  orientation: ReportOrientation;
}

export interface ReportJob {
  id: string;
  config: ReportConfig;
  status: ReportJobStatus;
  progress: number;
  createdAt: Date;
  completedAt?: Date;
  downloadUrl?: string;
}

// ── Backend API DTOs (mirror api/app/schemas/reports.py) ───────────────────

export type ApiReportTemplate = 'standard' | 'executive' | 'detailed';
export type ApiReportFormat = 'pdf';
export type ApiReportOrientation = 'portrait' | 'landscape';

export type ApiReportSectionId =
  | 'portfolio_summary'
  | 'weights'
  | 'performance_metrics'
  | 'risk_analytics';

export interface ApiReportBranding {
  company_name?: string;
  primary_color?: string;
  include_disclaimer?: boolean;
}

export interface ReportGenerateRequest {
  template?: ApiReportTemplate;
  sections: ApiReportSectionId[];
  branding?: ApiReportBranding;
  format?: ApiReportFormat;
  orientation?: ApiReportOrientation;
  portfolio_id?: string;
}

export interface ReportJobCreateResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface ReportJobProgress {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current: number;
  total: number;
  errors: string[];
  result: { report_id?: string; download_url?: string } | null;
  error: string | null;
}

export const REPORT_SECTIONS: ReportSection[] = [
  { id: 'summary', label: 'Executive Summary', description: 'Portfolio overview and KPIs', default: true },
  { id: 'performance', label: 'Performance Analysis', description: 'Returns, drawdowns, and benchmarks', default: true },
  { id: 'allocation', label: 'Asset Allocation', description: 'Current weights and sector breakdown', default: true },
  { id: 'risk', label: 'Risk Metrics', description: 'VaR, CVaR, volatility analysis', default: true },
  { id: 'attribution', label: 'Performance Attribution', description: 'Brinson and factor decomposition', default: false },
  { id: 'trades', label: 'Trade History', description: 'Recent rebalancing and transactions', default: false },
  { id: 'factors', label: 'Factor Exposures', description: 'Factor loadings and research signals', default: false },
  { id: 'appendix', label: 'Appendix', description: 'Methodology notes and data sources', default: false },
];
