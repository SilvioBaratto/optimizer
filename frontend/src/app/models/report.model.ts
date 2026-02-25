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
