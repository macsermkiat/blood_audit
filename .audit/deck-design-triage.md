# Deck hook triage

The hook reported eight unique finding groups in the existing standalone deck. A full static scan expanded them to 50 element findings. The preserved pre-documentation baseline already has all eight groups and 44 element findings. Longer decision-card text increases the tight-leading matches by six. None of the groups is treated as a newly introduced design system requirement.

| Finding group | Disposition | Evidence |
| --- | --- | --- |
| Tight leading, 1.28 | Fixed body descriptions and the compact route/outcome variant to 1.5. | Current CSS; computed decision and compact body line-height/font-size ratio is 1.5. |
| Tight leading, 1.25 | Fixed narrow report descriptions to 1.5. | At 390px, 12.48px text has 18.72px line height. |
| Tight leading, 1.26 | Fixed compact callout body text to 1.5. | At 1280 by 480, 12.288px text has 18.432px line height. |
| Cramped ledger rows | False positive from unresolved static clamp values; file-scoped rule exception. | Computed padding is 11.2px on narrow screens and 16.8px on desktop. |
| Cramped pipeline steps | False positive from unresolved static clamp values; same file-scoped rule exception. | Computed padding is 10.4px on narrow screens and 15.2px on desktop. |
| Cramped grounding layers | False positive from unresolved static clamp values; same file-scoped rule exception. | Computed padding is 11.2px on narrow screens and 16px on desktop. |
| Cramped slide wrappers | Corrected the actual footer inset, then retained a false-positive wrapper exception. | A two-value padding token was invalid as `left` or `right`. Explicit responsive insets now align the footer with the content at the verified 1440 by 900 desktop size. Shorter-screen content variants use different inset tokens. Other text sits inside `.slide-content`, which has real padding. |
| Cream palette | Intentional existing visual design; file-scoped palette exception. | Baseline warm paper tokens and the retained warm-neutral direction in PRODUCT.md and DESIGN.md. No palette redesign. |

Only the four body leading declarations and the footer's two horizontal inset declarations changed for this triage. Display-heading leading, fonts, navigation, clinical content, and the visual identity were preserved. The fixes are limited to the reported deck issues.

The two suppressions were persisted through the authorized `hooks ignore-value` command in `.impeccable/config.json`, scoped only to `clinical_audit_pitch_deck.html`. No `ignore-file`, `ignore-rule`, or project-wide suppression was used. Reasons attribute the judgment to the agent and observed/source evidence, not to a user confirmation.

Before suppression, the fixed deck has no tight-leading findings. The remaining 26 static findings consist of the 25 verified wrapper-padding false positives and the one intentional palette finding. After the two scoped exceptions, the detector returns zero findings. Full outputs are `deck-design-baseline.json`, `deck-design-current.json`, `deck-design-after-fixes.json`, and `deck-design-final.json`.

Rendered inspection covers actual 1440 by 900, 390 by 844, and 1280 by 480 viewport settings. At 390px, the decision grid has matching client and scroll heights of 711px and its final card ends about 795px into the 844px slide. At the short viewport, document width remains 1280px and compact body leading is 1.5. Original smooth navigation and viewport-resize settling were observed before reading the final bounds; intermediate positions are not final-fit evidence. Screenshots and final desktop measurements are saved beside this report.

These are local, uncommitted corrections. No application behavior or clinical policy was changed. The existing documentation verification remains applicable to the site; this standalone HTML-only follow-up does not require rebuilding the Astro source tree.
