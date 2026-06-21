import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { BottomTabBarComponent } from './bottom-tab-bar';
import { ICON_PROVIDER } from '../../icons';

describe('BottomTabBarComponent', () => {
  let fixture: ComponentFixture<BottomTabBarComponent>;

  beforeEach(async () => {
    await configureTestBed({
      imports: [BottomTabBarComponent],
      withRouter: true,
      withHttp: false,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(BottomTabBarComponent);
    fixture.detectChanges();
  });

  it('when rendered, it shows 3 routerLink navigation entries', () => {
    const links = fixture.nativeElement.querySelectorAll('a[routerLinkActive]');
    expect(links.length).toBe(3);
  });

  it('when rendered, each entry carries its accessible label', () => {
    const labels = Array.from(
      fixture.nativeElement.querySelectorAll('a[aria-label]'),
    ).map((a) => (a as HTMLElement).getAttribute('aria-label'));
    expect(labels).toEqual(['Dashboard', 'Portfolio', 'Optimize']);
  });
});
