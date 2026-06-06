import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { StepPersistPanelComponent } from './step-persist-panel';
import {
  StepStatus,
  type PersistStepResult,
} from '../../core/models/pipeline-builder.model';

const RESULT_PERSISTED: PersistStepResult = {
  persisted: true,
  reason: 'persisted',
  pass_count: 17,
  checklist_passed: true,
};

const RESULT_SKIPPED: PersistStepResult = {
  persisted: false,
  reason: 'persist_flag_off',
  pass_count: 17,
  checklist_passed: true,
};

const RESULT_FAILED: PersistStepResult = {
  persisted: false,
  reason: 'checklist_failed',
  pass_count: 14,
  checklist_passed: false,
};

function setup(): ComponentFixture<StepPersistPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection()],
  });
  return TestBed.createComponent(StepPersistPanelComponent);
}

function host(fix: ComponentFixture<StepPersistPanelComponent>): HTMLElement {
  return fix.nativeElement as HTMLElement;
}

describe('StepPersistPanelComponent', () => {
  describe('when stepStatus is Ready', () => {
    it('renders Run Step button with no form inputs', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('button[data-testid="run-step"]')).not.toBeNull();
      expect(el.querySelector('input,select,textarea,form')).toBeNull();
    });

    it('emits {} on click', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Ready);
      await fix.whenStable();

      const emitted: Record<string, unknown>[] = [];
      fix.componentInstance.runStep.subscribe((p) => emitted.push(p));

      host(fix)
        .querySelector<HTMLButtonElement>('button[data-testid="run-step"]')!
        .click();

      expect(emitted).toEqual([{}]);
    });
  });

  describe('Running state', () => {
    it('shows spinner, hides Run Step', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Running);
      await fix.whenStable();

      const el = host(fix);
      expect(el.querySelector('[data-testid="spinner"]')).not.toBeNull();
      expect(el.querySelector('button[data-testid="run-step"]')).toBeNull();
    });
  });

  describe('Completed render', () => {
    it('persisted badge shows "yes" when persisted', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_PERSISTED);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="persisted-badge"]');
      expect(badge).not.toBeNull();
      expect(badge!.textContent).toContain('yes');
    });

    it('persisted badge shows "no" when not persisted', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_SKIPPED);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="persisted-badge"]');
      expect(badge!.textContent).toContain('no');
    });

    it('renders reason and pass_count', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_PERSISTED);
      await fix.whenStable();

      const text = host(fix).textContent ?? '';
      expect(text).toContain('persisted');
      expect(text).toContain('17');
    });

    it('checklist badge shows "PASS" when checklist_passed is true', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_PERSISTED);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="checklist-badge"]');
      expect(badge).not.toBeNull();
      expect(badge!.textContent).toContain('PASS');
    });

    it('checklist badge shows "FAIL" when checklist_passed is false', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Completed);
      fix.componentRef.setInput('result', RESULT_FAILED);
      await fix.whenStable();

      const badge = host(fix).querySelector('[data-testid="checklist-badge"]');
      expect(badge!.textContent).toContain('FAIL');
    });
  });

  describe('Error render', () => {
    it('when lastError non-null, error block renders', async () => {
      const fix = setup();
      fix.componentRef.setInput('stepStatus', StepStatus.Error);
      fix.componentRef.setInput('lastError', 'Persist failed: DB error');
      await fix.whenStable();

      const errEl = host(fix).querySelector('[data-testid="error-block"]');
      expect(errEl).not.toBeNull();
      expect(errEl!.textContent).toContain('Persist failed: DB error');
    });
  });
});
