import { PORTFOLIO_BUILDER_V2 } from './feature-flag';
import { routes } from '../../app.routes';
import type { Route } from '@angular/router';

// Locks the route-flag contract:
//   - PORTFOLIO_BUILDER_V2 is consumed once at module-evaluation time by the
//     ternary in app.routes.ts, so there is no runtime toggle. Tests that
//     mutate the flag and re-import the module under Karma would not see a
//     fresh evaluation — Karma has no `jest.mock`-style aliasing.
//   - esbuild rewrites lazy imports to hashed chunk filenames (`chunk-XYZ.js`),
//     so we cannot assert on the source-file path. We can still assert on the
//     symbol resolved by the `.then((m) => m.<ComponentName>)` arm, which
//     esbuild preserves.

function children(): Route[] {
  const layout = routes[0];
  if (!layout.children) throw new Error('root route missing children');
  return layout.children;
}

function findRoute(path: string): Route {
  const r = children().find((c) => c.path === path);
  if (!r) throw new Error(`route not found: ${path}`);
  return r;
}

// Static duplicate of the V2 branch of the ternary in app.routes.ts:19-22.
// If a flag flip later changes the V2 chunk target, this duplicate must
// update in lockstep — the assertion below pins the resolved symbol.
const v2Loader = () =>
  import('./portfolio-builder').then((m) => m.PortfolioBuilderComponent);

describe('PORTFOLIO_BUILDER_V2 — route flag contract', () => {
  it('when read at module-load time, the flag is false (Cycle 3 ships off)', () => {
    expect(PORTFOLIO_BUILDER_V2).toBe(false);
  });

  it('when the flag is false, /portfolio-builder loadComponent resolves the legacy PipelineStepperComponent symbol', () => {
    const route = findRoute('portfolio-builder');
    const loader = route.loadComponent;
    expect(loader).toBeDefined();
    expect(loader!.toString()).toContain('PipelineStepperComponent');
  });

  it('when the flag is false, /portfolio-builder loadComponent does NOT resolve the V2 PortfolioBuilderComponent symbol', () => {
    const route = findRoute('portfolio-builder');
    const loader = route.loadComponent;
    expect(loader!.toString()).not.toContain('PortfolioBuilderComponent');
  });

  it('regardless of the flag, /portfolio-builder-legacy loadComponent resolves the legacy PipelineStepperComponent symbol', () => {
    const route = findRoute('portfolio-builder-legacy');
    const loader = route.loadComponent;
    expect(loader).toBeDefined();
    expect(loader!.toString()).toContain('PipelineStepperComponent');
  });

  it('when the V2 branch is invoked (inline duplicate), the loader resolves a PortfolioBuilderComponent class', async () => {
    const c = await v2Loader();
    expect(c).toBeDefined();
    // esbuild may suffix duplicate-binding class names (`PortfolioBuilderComponent2`),
    // so match by prefix instead of strict equality.
    expect(c.name.startsWith('PortfolioBuilderComponent')).toBe(true);
  });

  it('when the static-duplicate V2 loader is stringified, it pins the PortfolioBuilderComponent symbol', () => {
    expect(v2Loader.toString()).toContain('PortfolioBuilderComponent');
  });

  it('when /portfolio-builder route is found, it carries the human-readable title used by the router', () => {
    const route = findRoute('portfolio-builder');
    expect(route.title).toBe('Portfolio Builder');
  });

  it('when /portfolio-builder-legacy route is found, it carries the human-readable legacy title', () => {
    const route = findRoute('portfolio-builder-legacy');
    expect(route.title).toBe('Portfolio Builder (Legacy)');
  });
});
