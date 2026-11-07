---
name: ui-ux-guidelines
description: "Practical UI/UX implementation rules for layout, responsive design, touch targets, accessibility, and interaction patterns. Use when building or modifying frontend UI components — buttons, modals, forms, navigation, cards, or any interactive element. Covers the mechanical 'will this fit / work / be usable' questions, not creative aesthetics. Adapted from UI/UX Pro Max (github.com/nextlevelbuilder/ui-ux-pro-max-skill)."
---

# UI/UX Implementation Guidelines

Practical rules for building UI that works. Not about aesthetics — about
mechanics: will it fit, can they tap it, does it respond, does it break on
narrow screens.

**Think before typing.** Before writing layout code, answer:
1. What is the narrowest container this will render in?
2. How many items need to fit? Do they all fit at that width?
3. If they don't fit, what's the intentional fallback — stack, collapse, or scroll?

## Rule Priority

| Pri | Category | Level | Key Rule |
|-----|----------|-------|----------|
| 1 | Accessibility | CRITICAL | Contrast 4.5:1, focus rings, keyboard nav, ARIA labels |
| 2 | Touch & Interaction | CRITICAL | Min 44x44px targets, 8px gaps, feedback <150ms |
| 3 | Layout & Responsive | HIGH | Mobile-first, no overflow, container-aware sizing |
| 4 | Typography & Color | MEDIUM | 16px min body, 1.5 line-height, semantic tokens |
| 5 | Animation | MEDIUM | 150-300ms, transform/opacity only, respect reduced-motion |
| 6 | Forms & Feedback | MEDIUM | Visible labels, inline errors, loading states |
| 7 | Navigation | HIGH | Bottom nav max 5, predictable back, deep linking |

## 1. Accessibility (CRITICAL)

- Color contrast: 4.5:1 normal text, 3:1 large text
- Visible focus rings (2-4px) on all interactive elements — never remove
- Alt text for meaningful images; ARIA labels for icon-only buttons
- Full keyboard navigation with logical tab order
- Form labels paired with inputs — never placeholder-only
- Information via multiple channels, not color alone
- Sequential heading hierarchy (no skipping levels)
- Escape routes (cancel/back) in modals and multi-step flows

## 2. Touch & Interaction (CRITICAL)

- Touch targets: minimum 44x44px with 8px+ gaps between them
- Visual feedback on tap within 80-150ms (opacity, scale, or color shift)
- Disabled states: reduced opacity (0.38-0.5) + `cursor-not-allowed` + semantic attribute
- Loading indicator when async ops exceed 300ms
- Press feedback: subtle scale (0.95-1.05) or opacity change
- Never rely on hover-only interactions
- Never block user input during animations
- Never require pixel-perfect taps

### Drag and drop (the user's standard)

- The dragged item follows the cursor: a lifted copy of the card (shadow + accent ring, slight scale) rendered in a drag overlay (dnd-kit `DragOverlay`, high `zIndex`). Never a drag with no visible moving item.
- Source stays in place but dimmed; the valid drop target highlights (tint + inset ring) while hovered.
- Touch: press-and-hold activates the drag; a plain tap keeps its normal action. Escape cancels.
- Render each draggable exactly once. Responsive layouts reflow one node with grid placement; never a `hidden sm:flex` + `sm:hidden` duplicate (duplicate ids measure 0x0 and the overlay lands at the page corner).
- Drag tests in Playwright must intercept write requests before dragging, or they mutate real data.

## 3. Layout & Responsive (HIGH)

### Container-First Thinking

Before writing any layout:
- **Measure the container.** Know the actual rendered width (chat bubbles are
  ~350px, modals are ~500px, full-page is ~1200px).
- **Count the children.** If N items at minimum width > container width, they
  don't fit on one row. Plan the stack/wrap from the start.
- **Design the wrap state first** if wrapping is possible. The wrapped layout
  should look intentional, not accidental.

### Flex & Grid Decision Tree

- **All items fit on one row at min container width?** Use `flex` with no wrap.
- **Items might not fit?** Choose ONE:
  - `flex-wrap` — but design what the wrapped state looks like FIRST
  - Stacked rows — explicit multi-row layout (more predictable)
  - Grid with `auto-fit` / `auto-fill` — for uniform items
- **Never use `flex-wrap` without checking what the wrapped layout looks like.**
  Accidental wrap = orphaned items, clipped content, broken alignment.

### Rules

- Viewport meta: `width=device-width, initial-scale=1` (never disable zoom)
- Mobile-first: design for 375px, enhance upward
- Breakpoints: 375 / 768 / 1024 / 1440
- No horizontal scroll on mobile — content fits viewport width
- Min 16px body text on mobile (prevents iOS auto-zoom)
- Line length: 35-60 chars mobile, 60-75 chars desktop
- Spacing system: 4px/8px increments
- `min-h-dvh` instead of `100vh` on mobile
- Fixed nav/bars must reserve padding for underlying content
- Avoid nested scroll regions that fight main scroll
- Test at 375px AND landscape

### Anti-patterns

- Fixed `px` width containers
- `flex-wrap` without checking the wrapped layout
- `justify-between` on wrapping rows (orphans go to wrong edge)
- Hiding essential content on small screens without disclosure
- Assuming the container is wider than it is

## 4. Typography & Color (MEDIUM)

- Body text: 16px minimum, line-height 1.5-1.75
- Line length: 65-75 characters max
- Type scale: consistent steps (12/14/16/18/24/32)
- Font weight hierarchy: bold headings (600-700), regular body (400), medium labels (500)
- Semantic color tokens (`primary`, `error`, `surface`) — not raw hex in components
- Dark mode: desaturated/lighter tonal variants, not inverted colors
- Foreground/background pairs: 4.5:1 (AA) minimum
- Tabular/monospace figures for data columns, prices, timers
- Prefer wrapping over truncation; ellipsis only with tooltip

## 5. Animation (MEDIUM)

- Micro-interactions: 150-300ms; complex transitions max 400ms
- **Transform/opacity only** — never animate width/height/top/left (causes reflow)
- Ease-out for entering, ease-in for exiting
- Exit animations: 60-70% of enter duration
- List/grid stagger: 30-50ms between items
- Animations must be interruptible by user input
- Respect `prefers-reduced-motion`
- Max 1-2 animated elements per view
- Skeleton/shimmer for loading >300ms (not spinners)

## 6. Forms & Feedback (MEDIUM)

- **One editor per record.** Do not append a full record editor to an already-rendered form/configuration surface or keep parallel desktop/mobile field trees. Default full-form editing to the shared `EditorDialog` popup, reusing the existing fields and state owner. An inline edit must replace the displayed value/control in place, not duplicate an existing editor. Read-only summaries, immediate-save toggles, single-field replacement edits and navigation/workspace inspectors are not competing editors.
- Visible label with every input — never placeholder-only
- Error messages below the related field, not just form top
- Validate on blur, not keystroke
- Submit button shows loading then success/error state
- Required fields marked (asterisk)
- Confirmation dialog for destructive actions
- Toasts auto-dismiss 3-5s, don't steal focus
- Mobile input height: 44px minimum
- Semantic input types (`email`, `tel`, `number`) for correct keyboard
- After submit error: auto-focus first invalid field
- Undo option for destructive/bulk actions

## 7. Navigation (HIGH)

- Bottom nav: max 5 items with both icon and label
- Current location visually highlighted
- Predictable back behavior preserving scroll/state
- All key screens reachable via deep link
- Large screens (1024px+): prefer sidebar; small: bottom/top nav
- Single nav pattern per hierarchy level
- Modals are not for primary navigation
- Dangerous actions (delete, logout) visually and spatially separated

### Overlays and split screen (the user's standard)

BawtHub can pin a chat beside any app page (`WorkspaceSplitLayout`). The routed pane is a CSS container (`container-type: inline-size`), and that makes it the containing block for `position: fixed`. A `fixed inset-0` overlay rendered inline is therefore laid out inside the pane and under the chat (TASK-1407).

- **Never use an inline viewport overlay.** Anything that uses `position: fixed` to cover or position against the viewport (a modal, fullscreen view or viewport drawer) portals to `document.body`. New modals use `ViewportOverlay` (or `EditorDialog` for editors) from `@/lib/ui`, which portal for you. Keep the overlays that already portal.
- **Modal overlays** (backdrop, blocks the page): the backdrop covers the whole viewport, both panes included, and the panel centres on the viewport, not the pane. Escape and a backdrop click close it, and focus returns to the trigger.
- **Editor modal size (TASK-1407).** A modal that edits a record (many fields, code or JSON textareas) uses `EditorDialog` from `@/lib/ui`: up to `max-w-5xl` wide, the full viewport height less an even 1rem margin (no nav offset), header and footer pinned, only the body scrolls, full screen below `sm`. Fields that pair up sit in two columns; code textareas get 10+ rows. Short forms (a confirm with a few arguments) use `size="form"`: `max-w-2xl`, natural height, capped at the viewport. Plain confirms stay small (`max-w-sm`). Never cram an editor into a 500px card.
- **Expandable rows.** The whole row header is the toggle (a `button` with `aria-expanded` and a hover lift), not just the chevron. Row actions sit outside that button.
- **Dismissal guards override the close rule.** An in-flight action (a `working` or `sending` state), unsaved changes, or a nested widget that consumes Escape first (an open select, autocomplete or menu) may veto or defer the close. The guard has to be visible: a disabled close control or a confirm, never a silently ignored key.
- **Non-modal and anchored surfaces** (popovers, menus, slash-command lists, side drawers that share the page) keep their own anchoring and need no backdrop or viewport centring. They must still escape the pane's containing block when they position against the viewport, and close on Escape unless a guard above applies.
- Pages must work at the pane width as well as the viewport width. Check any change in split screen too, not only full screen.
- **Use container variants for page layout.** In the routed app tree, `@md:grid-cols-2` follows the pane width while `md:grid-cols-2` follows the viewport. The frontend test `splitScreenRatchet` fails if a file adds viewport-variant layout classes. A UI review includes a split-screen screenshot with chat pinned at a 1440 viewport.

## Pre-Delivery Checklist

Before shipping any UI change:

- [ ] Touch targets 44x44px with 8px gaps
- [ ] Tested at narrowest expected container width
- [ ] Wrapped/stacked states look intentional, not broken
- [ ] Text contrast 4.5:1 in both light and dark
- [ ] Interactive elements have visible focus rings
- [ ] Animations 150-300ms, interruptible, respect reduced-motion
- [ ] No horizontal overflow on mobile
- [ ] Loading states for async operations >300ms
- [ ] Form labels visible (not placeholder-only)
- [ ] Destructive actions have confirmation
- [ ] Drag shows a cursor-following card and a highlighted drop target
- [ ] Split screen, chat pinned beside the page at 1440: the page lays out at pane width, every modal it opens is portaled, centred on the viewport and above both panes, and anchored popovers stay attached to their trigger
- [ ] Modals close on Escape and on a backdrop click, with focus back on the trigger, unless a visible in-flight, unsaved or nested-widget guard applies
