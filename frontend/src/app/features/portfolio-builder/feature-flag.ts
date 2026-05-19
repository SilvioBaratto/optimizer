// Cycle 4 flips this ON after live-state UX parity verification.
// The `as boolean` cast prevents TypeScript from narrowing the literal to
// `false`, which would dead-code-eliminate the new shell branch in the
// router ternary and stop the new lazy chunk from emitting / type-checking.
export const PORTFOLIO_BUILDER_V2 = true as boolean;
