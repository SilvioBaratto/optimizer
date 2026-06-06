import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepSectionComponent } from './step-section';
import { StepStatus } from '../../models/pipeline-builder.model';

describe('StepSectionComponent', () => {
  let fixture: ComponentFixture<StepSectionComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideZonelessChangeDetection()],
    });
  });

  function build(
    inputs: Partial<{
      title: string;
      description: string;
      status: StepStatus;
      summary: string;
      expanded: boolean;
      showContinue: boolean;
      showRetry: boolean;
    }> = {},
  ): StepSectionComponent {
    fixture = TestBed.createComponent(StepSectionComponent);
    const ref = fixture.componentRef;
    ref.setInput('title', inputs.title ?? 'Load');
    ref.setInput('description', inputs.description ?? '');
    ref.setInput('status', inputs.status ?? StepStatus.Ready);
    ref.setInput('summary', inputs.summary ?? '');
    ref.setInput('expanded', inputs.expanded ?? false);
    ref.setInput('showContinue', inputs.showContinue ?? false);
    ref.setInput('showRetry', inputs.showRetry ?? false);
    fixture.detectChanges();
    return fixture.componentInstance;
  }

  function html(): string {
    return (fixture.nativeElement as HTMLElement).innerHTML;
  }

  describe('header toggle', () => {
    it('when the header button is clicked, toggle emits', () => {
      const c = build();
      let toggled = 0;
      c.toggle.subscribe(() => (toggled += 1));
      (fixture.nativeElement as HTMLElement)
        .querySelector('button')!
        .click();
      expect(toggled).toBe(1);
    });
  });

  describe('badge label', () => {
    it('renders a label per StepStatus', () => {
      build({ status: StepStatus.Ready });
      expect(html()).toContain('Ready');
      build({ status: StepStatus.Running });
      expect(html()).toContain('Running');
      build({ status: StepStatus.Completed });
      expect(html()).toContain('Done');
      build({ status: StepStatus.Error });
      expect(html()).toContain('Failed');
      build({ status: StepStatus.Locked });
      expect(html()).toContain('Locked');
    });
  });

  describe('title + description', () => {
    it('renders the elegant title, not an id or number', () => {
      build({ title: 'Clean returns' });
      expect(html()).toContain('Clean returns');
    });

    it('shows the description when collapsed and there is no summary', () => {
      build({
        expanded: false,
        summary: '',
        description: 'Validate, winsorize and impute the return series',
      });
      expect(html()).toContain(
        'Validate, winsorize and impute the return series',
      );
    });

    it('summary replaces description when collapsed and summary present', () => {
      build({
        expanded: false,
        summary: '2,902 tickers · 1,294 days · EUR',
        description: 'Fetch price & macro history',
      });
      expect(html()).toContain('2,902 tickers · 1,294 days · EUR');
      expect(html()).not.toContain('Fetch price & macro history');
    });
  });

  describe('collapsed summary', () => {
    it('shows the summary when collapsed and non-empty', () => {
      build({ expanded: false, summary: '612 tickers · 2519 days · EUR' });
      expect(html()).toContain('612 tickers · 2519 days · EUR');
    });

    it('hides the summary when expanded', () => {
      build({ expanded: true, summary: '612 tickers · 2519 days · EUR' });
      expect(html()).not.toContain('612 tickers · 2519 days · EUR');
    });

    it('renders no summary span when summary is empty', () => {
      build({ expanded: false, summary: '' });
      // only the header button exists, no projected body
      expect(html()).not.toContain('text-text-secondary truncate');
    });
  });

  describe('body + actions', () => {
    it('renders the body only when expanded', () => {
      build({ expanded: false });
      expect(html()).not.toContain('border-t border-border');
      build({ expanded: true });
      expect(html()).toContain('border-t border-border');
    });

    it('Continue button visible only when expanded + showContinue', () => {
      build({ expanded: true, showContinue: true });
      expect(html()).toContain('Continue →');
      build({ expanded: false, showContinue: true });
      expect(html()).not.toContain('Continue →');
    });

    it('Retry button visible only when expanded + showRetry', () => {
      build({ expanded: true, showRetry: true });
      expect(html()).toContain('Retry');
    });

    it('continueStep / retry emit on click', () => {
      const c = build({ expanded: true, showContinue: true, showRetry: true });
      let cont = 0;
      let ret = 0;
      c.continueStep.subscribe(() => (cont += 1));
      c.retry.subscribe(() => (ret += 1));
      const buttons = (
        fixture.nativeElement as HTMLElement
      ).querySelectorAll('button');
      // [0] header, [1] Continue, [2] Retry
      (buttons[1] as HTMLButtonElement).click();
      (buttons[2] as HTMLButtonElement).click();
      expect(cont).toBe(1);
      expect(ret).toBe(1);
    });
  });
});
