import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { configureTestBed, setInput } from './index';

@Component({
  template: '<span>{{ label() }}</span>',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
class HarnessHost {
  readonly label = input.required<string>();
}

describe('setInput', () => {
  it('when a required signal input is set, the template reflects it after CD', async () => {
    await configureTestBed({ imports: [HarnessHost], withHttp: false });
    const fixture = TestBed.createComponent(HarnessHost);

    setInput(fixture, 'label', 'hello');

    expect(fixture.componentInstance.label()).toBe('hello');
    expect(fixture.nativeElement.textContent).toContain('hello');
  });
});
