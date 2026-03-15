import { Component, signal, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { ModalService } from './modal.service';
import { ReportGeneratorService } from '../../services/report-generator.service';
import {
  REPORT_SECTIONS,
  type ReportTemplate,
  type ReportFormat,
  type ReportOrientation,
} from '../../models/report.model';

interface TemplateOption {
  value: ReportTemplate;
  label: string;
}

const TEMPLATE_OPTIONS: TemplateOption[] = [
  { value: 'executive_summary', label: 'Executive Summary' },
  { value: 'full_report', label: 'Full Report' },
  { value: 'risk_report', label: 'Risk Report' },
  { value: 'performance_attribution', label: 'Performance Attribution' },
];

@Component({
  selector: 'app-export-report-modal',
  imports: [LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="space-y-5">

      <!-- Template -->
      <div>
        <label class="block text-sm font-medium text-text mb-1.5" for="report-template">
          Report Template
        </label>
        <select
          id="report-template"
          class="w-full px-3 py-2 bg-surface-inset border border-border rounded-lg text-sm text-text focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent transition-colors"
          [value]="template()"
          (change)="template.set($any($event.target).value)">
          @for (opt of templateOptions; track opt.value) {
            <option [value]="opt.value">{{ opt.label }}</option>
          }
        </select>
      </div>

      <!-- Sections -->
      <div>
        <p class="text-sm font-medium text-text mb-2">Report Sections</p>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
          @for (section of reportSections; track section.id) {
            <label class="flex items-start gap-2.5 cursor-pointer group">
              <input
                type="checkbox"
                class="mt-0.5 w-4 h-4 rounded border-border text-accent focus:ring-accent/40 focus:ring-2 bg-surface-inset cursor-pointer"
                [checked]="isSectionSelected(section.id)"
                (change)="toggleSection(section.id)">
              <div class="min-w-0">
                <p class="text-sm text-text group-hover:text-text leading-tight">{{ section.label }}</p>
                <p class="text-xs text-text-tertiary mt-0.5 leading-tight">{{ section.description }}</p>
              </div>
            </label>
          }
        </div>
      </div>

      <!-- Branding -->
      <div>
        <p class="text-sm font-medium text-text mb-2">Branding</p>
        <div class="space-y-3">
          <div>
            <label class="block text-xs text-text-secondary mb-1" for="company-name">Company Name</label>
            <input
              id="company-name"
              type="text"
              class="w-full px-3 py-2 bg-surface-inset border border-border rounded-lg text-sm text-text placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent transition-colors"
              placeholder="Your Company Name"
              [value]="companyName()"
              (input)="companyName.set($any($event.target).value)">
          </div>
          <div class="flex items-center gap-3">
            <div class="flex-1">
              <label class="block text-xs text-text-secondary mb-1" for="primary-color">Primary Color</label>
              <div class="flex items-center gap-2">
                <input
                  id="primary-color"
                  type="color"
                  class="w-9 h-9 rounded-lg border border-border bg-surface-inset cursor-pointer p-0.5"
                  [value]="primaryColor()"
                  (input)="primaryColor.set($any($event.target).value)">
                <span class="text-sm font-mono text-text-secondary">{{ primaryColor() }}</span>
              </div>
            </div>
            <label class="flex items-center gap-2 cursor-pointer self-end pb-1">
              <input
                type="checkbox"
                class="w-4 h-4 rounded border-border text-accent focus:ring-accent/40 focus:ring-2 bg-surface-inset cursor-pointer"
                [checked]="includeDisclaimer()"
                (change)="includeDisclaimer.set($any($event.target).checked)">
              <span class="text-sm text-text">Include Disclaimer</span>
            </label>
          </div>
        </div>
      </div>

      <!-- Format + Orientation -->
      <div class="grid grid-cols-2 gap-4">
        <div>
          <p class="text-sm font-medium text-text mb-2">Format</p>
          <div class="space-y-1.5">
            @for (fmt of formatOptions; track fmt.value) {
              <label class="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="report-format"
                  class="w-4 h-4 border-border text-accent focus:ring-accent/40 focus:ring-2 bg-surface-inset cursor-pointer"
                  [value]="fmt.value"
                  [checked]="format() === fmt.value"
                  (change)="format.set(fmt.value)">
                <span class="text-sm text-text">{{ fmt.label }}</span>
              </label>
            }
          </div>
        </div>
        <div>
          <p class="text-sm font-medium text-text mb-2">Orientation</p>
          <div class="space-y-1.5">
            @for (ori of orientationOptions; track ori.value) {
              <label class="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="report-orientation"
                  class="w-4 h-4 border-border text-accent focus:ring-accent/40 focus:ring-2 bg-surface-inset cursor-pointer"
                  [value]="ori.value"
                  [checked]="orientation() === ori.value"
                  (change)="orientation.set(ori.value)">
                <span class="text-sm text-text">{{ ori.label }}</span>
              </label>
            }
          </div>
        </div>
      </div>

      <!-- Validation message -->
      @if (selectedSections().length === 0) {
        <p class="text-xs text-warning">Select at least one section to generate a report.</p>
      }

      <!-- Actions -->
      <div class="flex justify-end gap-3 pt-2 border-t border-border">
        <button
          type="button"
          class="px-4 py-2 text-sm font-medium text-text-secondary border border-border rounded-lg hover:bg-surface-inset transition-colors"
          (click)="onCancel()">
          Cancel
        </button>
        <button
          type="button"
          class="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium bg-accent text-white rounded-lg hover:bg-accent/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          [disabled]="cannotGenerate()"
          (click)="onGenerate()">
          <i-lucide name="download" />
          Generate Report
        </button>
      </div>

    </div>
  `,
})
export class ExportReportModalComponent {
  private readonly modalService = inject(ModalService);
  private readonly reportGenerator = inject(ReportGeneratorService);

  readonly templateOptions = TEMPLATE_OPTIONS;
  readonly reportSections = REPORT_SECTIONS;

  readonly formatOptions: Array<{ value: ReportFormat; label: string }> = [
    { value: 'pdf', label: 'PDF' },
    { value: 'xlsx', label: 'Excel (XLSX)' },
    { value: 'csv', label: 'CSV' },
  ];

  readonly orientationOptions: Array<{ value: ReportOrientation; label: string }> = [
    { value: 'portrait', label: 'Portrait' },
    { value: 'landscape', label: 'Landscape' },
  ];

  readonly template = signal<ReportTemplate>('executive_summary');
  readonly selectedSections = signal<string[]>(
    REPORT_SECTIONS.filter(s => s.default).map(s => s.id),
  );
  readonly companyName = signal('');
  readonly primaryColor = signal('#6366f1');
  readonly includeDisclaimer = signal(true);
  readonly format = signal<ReportFormat>('pdf');
  readonly orientation = signal<ReportOrientation>('portrait');

  readonly cannotGenerate = computed(() => this.selectedSections().length === 0);

  isSectionSelected(id: string): boolean {
    return this.selectedSections().includes(id);
  }

  toggleSection(id: string): void {
    this.selectedSections.update(current =>
      current.includes(id)
        ? current.filter(s => s !== id)
        : [...current, id],
    );
  }

  onGenerate(): void {
    if (this.cannotGenerate()) return;

    this.reportGenerator.generateReport({
      template: this.template(),
      sections: this.selectedSections(),
      branding: {
        companyName: this.companyName(),
        primaryColor: this.primaryColor(),
        includeDisclaimer: this.includeDisclaimer(),
      },
      format: this.format(),
      orientation: this.orientation(),
    });

    this.modalService.close();
  }

  onCancel(): void {
    this.modalService.close();
  }
}
