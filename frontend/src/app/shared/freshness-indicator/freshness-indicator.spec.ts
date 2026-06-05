import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from '../../../testing';
import { FreshnessIndicatorComponent } from './freshness-indicator';
import type { FreshnessLevel } from '../../models/jobs.model';

describe('FreshnessIndicatorComponent', () => {
  let fixture: ComponentFixture<FreshnessIndicatorComponent>;
  let comp: FreshnessIndicatorComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [FreshnessIndicatorComponent], withHttp: false });
    fixture = TestBed.createComponent(FreshnessIndicatorComponent);
    comp = fixture.componentInstance;
  });

  const cases: ReadonlyArray<[FreshnessLevel, string, string]> = [
    ['fresh', 'bg-gain', 'text-gain'],
    ['stale', 'bg-warning', 'text-warning'],
    ['critical', 'bg-loss animate-pulse', 'text-loss'],
    ['unknown', 'bg-text-tertiary', 'text-text-tertiary'],
  ];

  for (const [level, dot, label] of cases) {
    it(`when level is ${level}, the dot and label classes match`, () => {
      setInput(fixture, 'level', level);
      expect(comp.dotClass()).toBe(dot);
      expect(comp.labelClass()).toBe(label);
    });
  }

  it('when level is unknown, the label reads "No data"', () => {
    setInput(fixture, 'level', 'unknown');
    expect(comp.displayLabel()).toBe('No data');
  });

  it('when an age label is given, it is shown instead of the raw level', () => {
    setInput(fixture, 'level', 'fresh');
    setInput(fixture, 'ageLabel', '2m ago');
    expect(comp.displayLabel()).toBe('2m ago');
  });

  it('when no age label is given, the level is shown', () => {
    setInput(fixture, 'level', 'stale');
    expect(comp.displayLabel()).toBe('stale');
  });
});
