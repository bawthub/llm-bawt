# Design elements — icon systems, launcher tiles, floating glass surfaces

> Part of the bawthub skill. Read before touching home/tools launcher tiles, nav icons, any floating surface (drawer/popover/overlay/menu), or any UI that picks/shows a bot. The #1 visual regression is picking the wrong icon treatment.

### Shared notifications — one system

**All user-facing notifications use the shared notification system unless the user
explicitly approves an exception.** Do not add component-local toast state,
portals, stacks, dismissal/expiry timers, or a second notification library.
Refactor an existing custom result popup through the shared system when touching
that notification; do not copy its separate lifecycle into another component.

Use `notify` from `@/app/notifications` (`notify.success`, `.warning`, `.error`,
`.info`; `notify.update` for an evolving result). The shared
`notifications/useNotificationStore.ts` and global `NotificationHost.tsx` own
presentation, lifetime, deduplication and Events history. Pass the existing
`action` option for task links/buttons and the appropriate tone and duration;
keep queued, confirmed and failed outcomes distinct. Do not show a second toast
for an event already surfaced by its shared handler.

A local submission confirmation needs no invented backend event or persistent
bell entry. `task_changed` refreshes task data; it is not a notification receipt.
For durable system events and established specialized surfaces (ops, approvals,
inter-bot relay pills), reuse their existing event handlers and shared history
integration rather than adding a parallel notification path. Form-field
validation and ordinary inline content are not toast notifications.

### Design elements — the three icon systems

BawtHub uses **three** distinct icon treatments. Picking the wrong one is the
most common visual regression; they are not interchangeable.

| Surface | Treatment | Source of truth |
|---|---|---|
| **Launcher tiles** — home + `/tools` grid | **Color emoji**, no frame | `HomeClient.tsx` `TOP_TILES`/`BOTTOM_TILES`/`EXTRA_TILES`, `tools/page.tsx` `TOOLS_MENU_ITEMS` |
| **Nav pills / sub-nav** — `AppMainNav`, `SecondaryNav` | lucide stroke icon + per-tone color | `AppMainNav.tsx` `TONE_ICON_COLORS` |
| **In-page controls** — buttons, rows, menus | lucide stroke icon, `white/60`–`white/90` | component-local |

**Launcher tiles are emoji, not line icons.** When "main menu" / "home tiles"
comes up, the user means the **emoji launcher**, not the nav pills. Twelve thin
monochrome strokes at tile size read as one texture and force you to read the
label to find anything — which defeats a launcher. A color pictograph is
identifiable pre-attentively. Do NOT "upgrade" a launcher to lucide icons, and
do not wrap the emoji in a dark disc/badge (framing only shrinks it; the disc
existed to give a thin stroke something to read against).

Launcher tile recipe (copy it, don't reinvent):
- emoji at `text-5xl`→`text-7xl`, `select-none`, `filter: drop-shadow(0 4px 12px rgba(0,0,0,0.4))`
- hover: `scale(1.1–1.12)` + `-rotate-3`; per-tile hover float keyframes live in
  `home-clock.css` (`float-<label>`), gated behind
  `@media (hover: hover) and (prefers-reduced-motion: no-preference)` — an
  idle `infinite` animation over `backdrop-blur` pinned the page at 60fps, so
  **hover-only, never idle**
- card: `rounded-2xl border border-white/10 bg-slate-900/60 backdrop-blur-sm`,
  a per-tile `gradient` wash at `opacity-0 group-hover:opacity-[0.07]`, and a
  `glow`/`hoverGlow` shadow pair
- press feedback is app-wide: tapped tile scales up + spinner, siblings dim to
  `opacity-30`, then navigate on a ~240–350ms timer

**Reuse the same emoji across launchers for the same concept** (💬 Chat, 🐳
Docker, 🧑‍🎤 Avatar appear on both home and `/tools`) so the two launchers teach
each other. Within one grid, make every glyph differ in **both silhouette and
dominant color** — no two whitish documents side by side.

**Home must mirror `AppMainNav`.** Home tiles are the launcher for the same
destinations as `NAV_ITEMS`; when a route is added to the nav, add a tile too
(this drifted once — Repos and Studio were nav-only and invisible on home).
Rows are `3 / 3 / 2` via a single `NavTileCard` with a `TILE_SIZES` variant —
add to an existing array rather than copying markup for a new row.

### Bot display suite — bots are never a plain list

Bots are the product, so every place a bot is chosen or shown gets the full
identity treatment (avatar, palette color, agent/chatbot grouping). A text-only
bot dropdown is a design violation, even in an admin form. All components live in
`frontend/src/app/chat/`:

| Need | Use | Notes |
|---|---|---|
| Choose a bot | `BotSelect` | Glass popover with avatar rows. `groupByKind` splits Chatbots/Agents, `placeholder`, `size="md"` + `className="w-full justify-between"` for form fields. Inside a modal, raise `zClassName` above the modal layer (e.g. `z-[1100]` over a `z-[1000]` dialog). |
| Show "this bot" (name + avatar) | `BotChip` | One indivisible unit; variants `tint`/`outline`/`solid`/`plain`. Never hand-wire `BotAvatar` next to colored text. |
| Avatar alone | `BotAvatar` | Self-hosted render → emoji → letter. Never fetches from a CDN. |
| Picker/rail tile identity | `BotPickerAvatar` | Aura, selection rim, activity pip. |
| Message author | `MessageAuthorHeader` | |
| Data | `useBots()` → `BotRecord[]`, `useBotColors()` + `getPalette()` | One cached query. Never add a private `/v1/bots` fetch with a slimmed `{slug,name}` type, because it strips avatar/color/agent_backend and forces a plain list. |

**Size: big bots in work surfaces.** Where bots are the content (assignment boards, model/tag rosters, drag targets) use `BotChip size="md"` (13px+ name, readable avatar). `xs`/`sm` are only for dense inline metadata. Data-dense settings pages (e.g. `/tools/settings/models`) are full width (`isWide` in `settings/layout.tsx`); no centered `max-w-5xl` column leaving dead space. Moving a bot = drag the chip with the overlay rules in `ui-ux-guidelines` § Drag and drop (reference: `tools/botModelDrag.tsx`).

**Bot-to-bot feedback:** `notifications/InterBotRelay.tsx` composes `BotChip`s into the delivery pill; `BotRelayPulse.tsx` adds send/receive effects to rail avatars. Both share the delivery handler and `useRelayParty()` identity/color resolution. See [inter-bot-relay.md](inter-bot-relay.md) for lifecycle rules, expiry ownership, motion, rail clipping constraints and verification.

**Banned for bots:** native `<select>`/`<option>`, `@/lib/ui` `Select` with
`bots.map(b => ({value, label}))`, bare name text lists.

**Bot-derived defaults:** anything created *for* a bot defaults to that bot's
identity. Example: Home/Tools nav tiles added from the Bots library take the bot's
avatar emoji (`botEmoji()` in `settings/navigation/editorState.ts`, which uses
`isEmoji` from `BotAvatar.tsx`; image avatars fall back to the catalog emoji).

Sweep before shipping bot UI:
`grep -rniE "<option[^>]*(bot|\.slug)" src --include=*.tsx` and
`grep -rn "bots\.map((\w*) => ({ value" src`. Both should return nothing.

### TaskTag — task identity is its own tag (TASK-1528)

A task's short ID (`TASK-N`) is its identity and gets a single, consistent
treatment everywhere a task appears. Components are in
`frontend/src/app/tasks/TaskTag.tsx`:

| Need | Use | Notes |
|---|---|---|
| Show which task this is | `TaskTag` | Mono sky pill. It looks the same in every status. Use `size="xs"` (default) everywhere it sits beside badges, and `"sm"` only for a standalone larger heading. Pass `children` to render a search-highlighted ID. |
| Show the task's state | `TaskStatusBadge` | `STATUS_META` badge + icon + **text label**. Status is never shown by colour alone. |

Rules:
- **Order:** TaskTag first, then the title, then the status badge and metadata. Never merge ID and status into one pill.
- **No repeats:** a row that shows the TaskTag and status badge must not repeat `TASK-N` or the status in its subtitle or snippet. Spotlight's `toTaskSearchResult()` subtitle is the outcome or description only, and its badge is the project only.
- **Don't shrink it away:** the tag is `shrink-0 whitespace-nowrap`. Long titles wrap or clamp instead, so the ID stays readable at 375px.
- **Not a bot tag:** bot taxonomy tags are a separate system (`BotChip` and tag pills). Don't reuse TaskTag colours for them.
- **Where it's used:** Spotlight task results, the task modal header (`TaskDetailView`, whose `pr-8` keeps the badges clear of the modal's ×), and `AllTasksMobileList`. New task surfaces should use it rather than a gray mono span.

### Section bar — page controls live in the third menu (TASK-1382)

The section bar is the third row of chrome: app nav, then the section bar
(`SecondaryNav`, fed by `MainMenuSubNav`), then the page. It holds the
section's tabs and, after a divider, the current page's own controls: view
switches, search, filters, sort, counts and page tools such as Columns or
New. One row, never wrapped. It folds by measured width (labels to icons,
inline pickers into one Filters menu, search to an icon, tabs into a picker),
and once every fold step is spent the row scrolls sideways.

Rules:
- **Tabs read as menus.** Every tab-style menu has a visibly darker, recessed
  track, not a transparent row of labels. `globals.css` applies the shared
  `--tab-track` treatment to `[role="tablist"]`; segmented menus without tab
  semantics use `tab-menu-track`. Keep selected tabs visibly raised/contrasting,
  focus rings visible, and narrow tab rows intentionally scrollable.
- **No page-level toolbar in the page body.** Don't put a filter row,
  search box or view toggle above the content, in a `PageHeader` action slot,
  or as a separate phone-only filter panel. Register the page's controls in
  its section's toolbar map instead (`TasksToolbar.tsx` `TOOLBARS`:
  `{ label, Controls, foldSteps }`). Controls inside one card, row or pane
  stay with that card, row or pane. Exception: a page migrated to the
  three-layer model (below) puts its views and actions in `ViewBar` at the top
  of its content.
- **State the bar can reach.** The layout renders the bar, so page-option
  state lives in the URL or a small store shared with the page, never in the
  page component's local state (Board: URL + prefs; All Tasks:
  `all/allTasksView.ts`).
- **Fold, then scroll.** Read the current step with `useSectionBarFold()` and
  give each step one change. Never wrap to a second row or squeeze a control
  below its 32px box (`touch-hit` supplies the 44px tap area).
- **Reuse the bar's controls:** `CONTROL`, `toolButton` and `SearchControl`
  from `tasks/board/BoardViewControls.tsx`, `Select size="sm"`, `BotSelect`
  for agents, and `AnchoredPopover` + `floating-glass-surface` for folded
  menus.
- **Examples:** `/tasks` (Relay), `/tasks/board`, `/tasks/all`, Repos
  (`RepoWorkspaceChrome`). Tools has no page-controls hook yet, so Tools
  pages that still have in-page toolbars are tracked as follow-up cards.

### Three layers and the fold rule (TASK-2173)

Navigation has three layers, one component each:

1. **Sections**: the app top bar (`AppMainNav`).
2. **Pages**: the section bar (`SecondaryNav` via `MainMenuSubNav`).
3. **Views** of one page: `ViewTabs` (`lib/ui/ViewTabs.tsx`). It sits in its
   own row at the top of the page content, on the shared tab track, and is never
   put in the section bar.

Page actions such as refresh and new go in `PageToolbar`
(`lib/ui/PageToolbar.tsx`). It sits to the right of the views; `ViewBar` lays
out both, and on a phone the toolbar takes its own row.

**Fold rule:** only the toolbar folds, first from labels to icons, then into one
"⋯" menu. Views never become a dropdown. A narrow view row scrolls sideways,
fades its clipped edge and keeps the active view in sight.

Rules:
- `ViewTabs` is real ARIA tabs over one `ViewTabPanel`. Arrow keys, Home and End
  move focus; Enter or Space opens the view. Views that are separate routes keep
  link semantics, not tabs.
- Hold the view in the URL with `useUrlView(param, keys, fallback, { clear })`
  (`lib/useUrlView.ts`), so Back, Forward, reload and deep links all work.
  `clear` drops params that belong to the previous view.
- Proof page: `/tools/approvals` (`?tab=`). Other pages still put their views in
  the section bar until they migrate. Don't change the look of the top bar or
  the section bar for this.

### Floating glass surfaces — reuse, do not recreate

Drawers, popovers, floating inspectors, docked overlays, and transient menus use
`floating-glass-surface` from `frontend/src/app/globals.css`. It intentionally
matches the top-menu glass: cyan edge, 40–48% dark fill, 12px backdrop frost,
inset highlight, and restrained shadow. Apply geometry separately:

```tsx
<div className="floating-glass-surface rounded-xl border">
  {/* rows/content must stay transparent or lightly tinted */}
</div>
```

Rules:
- **Do not copy the gradient/filter/shadow inline.** Reuse the class; change the
  shared class when the product-wide floating treatment changes.
- Keep direct child surfaces `bg-transparent` or very lightly tinted. An opaque
  `bg-slate-900/*` child paints over the backdrop and kills visible bleed-through.
- Avoid `transform` on ancestors of backdrop-filtered UI. A transformed wrapper
  creates a stacking context that can disable or corrupt descendant backdrop blur
  in Chromium; position with `top/left/right/bottom` when possible.
- Do not nest backdrop filters. Blur once on the outer floating surface; inner
  rows use translucent fills/highlights only. Nested filters add GPU work and can
  produce white fringes while content streams underneath.
- Use solid `bg-slate-900/95` only when transparency would hurt legibility (dense
  forms, destructive confirmations, or full-screen work surfaces), not as the
  default for floating chrome.

### Workspace surfaces — the Repos page pattern (TASK-918)

Reference implementation for dense, multi-pane work surfaces (file lists, diff
documents, grouped rails): `repos/page.tsx`, `repos/workspace/GroupedFileRail.tsx`,
`diffs/DiffDocumentSection.tsx`, `diffs/ContinuousDiffViewer.tsx`. The user iterated
on this directly; these are his calls, not taste to relitigate.

**Separation = wireframe + space, never colored rails.** Distinguish items (a file,
a group) with a thin faint-sky outline box plus a gap of page background:
`rounded-lg border border-sky-200/[0.14]` + `mt-2`/`mt-4`. Rejected: colored
left-edge rails by change type, glows, drop-shadow cards, neutral hairline dividers.

**Panes are inset rounded cards, not flush rectangles.** Each pane gets
`WORKSPACE_PANE` (`overflow-hidden rounded-xl border border-sky-200/[0.12]`
+ soft `0 6px 18px` shadow) inside a `p-1.5 gap-0.5` flex row. The resize
separator's line is transparent at rest and appears only on hover/focus — a
permanent line in the gap between two cards reads as boxy.

**Glass lives on sticky chrome only.** Sticky group headers and the diff's
floating position bar use `floating-glass-surface` so content blurs beneath
while scrolling. A floating sticky bar is inset (`sticky top-2 mx-2 mt-2 sm:mx-3
rounded-lg border`), aligned with the cards below, radius matching them. When a
sticky element gets a `top` offset, scroll-landing math must add it
(`useDiffDocumentNavigation` `stickyOffset`) or targets tuck under the bar.
Document headers (the per-file band) stay mostly solid for contrast against code:
`bg-slate-800/70` + `from-sky-300/[0.08]` sheen + inset top highlight.

**"Selected/active" = background lift, never a border.** The user: *the effect is
applied to the background within the element* — do not add solid/bright borders
or rings to show state. Active group/file: `bg-sky-400/[0.055]` (he halved it
from 11%; subtle is the target). Siblings of an active scope dim to
`opacity-70 hover:opacity-100` — the dimming does most of the work. Derive
"active" from the page's single source of truth (`scope.groupId`), never a
component-local copy.

**Clickable headers signal with a hover lift, not an icon.** A header title that
opens something (task modal, `/chat/<bot>/<session>`) is marked
`data-title-action`; the header's `::before` overlay (`bg-sky-200/[0.08]`) fades
in via `has-[[data-title-action]:hover]`. Rejected: a separate "open" (⤢) icon
button. Split hit areas: chevron collapses, title acts.

**Scrub, don't hover-navigate.** Press-and-drag over the file rail scrolls the
diff to the row under the pointer (`useRailScrub`), with edge auto-scroll and one
history entry per drag; a press without movement is still a click. Plain hover
must never navigate — crossing rows en route to the diff would jump around.
Mouse only; touch keeps native scroll.

## UI hot rules (moved from SKILL.md, TASK-1640)

Building/modifying UI components → load the **`ui-ux-guidelines`** skill (container-first thinking, flex/grid decision tree). Design treatments → **`references/design-system.md`** before touching launcher tiles, nav icons, or floating surfaces. Hot rules:

- **Three icon systems, not interchangeable:** launcher tiles (home + `/tools`) = **color emoji, no frame** (when the user says "main menu"/"home tiles" he means the emoji launcher — do NOT "upgrade" to lucide or add a disc); nav pills = lucide + per-tone color (`AppMainNav.tsx`); in-page controls = lucide `white/60`–`white/90`.
- **Which "menu" the user means:** "Tools menu" = the `/tools` launcher tiles (`TOOLS_MENU_ITEMS` in `tools/page.tsx`). The **top menu** (`AppMainNav`) is fixed `NAV_ITEMS` + the user's toggleable favorites (bookmark icon / Settings → Appearance) — never hardcode extra links there (TASK-866 wrongly pinned Approvals/Models; removed in TASK-940).
- **Page controls go in the section bar (the third menu).** Filters, search, view switches and page tools sit after the section tabs on one row that folds, then scrolls sideways. No in-page toolbar rows or phone-only filter panels. See `references/design-system.md` § Section bar.
- **Floating surfaces reuse `floating-glass-surface`** (globals.css) — never copy the gradient/filter inline; children stay transparent; no `transform` on ancestors of backdrop-filtered UI; never nest backdrop filters.
- **Home tiles must mirror `AppMainNav` `NAV_ITEMS`** — new nav route ⇒ new tile.
- **Bots are never a plain list.** Bots are the product: every place a bot is chosen or shown uses the shared bot-display suite in `src/app/chat/` — choose = `BotSelect` (avatar + palette, `groupByKind`, `zClassName` inside modals), label = `BotChip`, avatar = `BotAvatar`/`BotPickerAvatar`, data = `useBots()` (`BotRecord`) + `useBotColors()`. **Never** a native `<select>`, `@/lib/ui` `Select`, or a private `/v1/bots` query for bots. Bot-specific defaults come from the bot too (e.g. a new bot tile's emoji = the bot's avatar). Inventory: `references/design-system.md` § Bot display suite.
- **Task identity = `TaskTag`.** Any `TASK-N` shown on a task surface uses `TaskTag` from `src/app/tasks/TaskTag.tsx`, placed first with a separate `TaskStatusBadge` (text label, not colour alone). Never put a gray mono ID span there, and never repeat the ID or status in the row snippet. See `references/design-system.md` § TaskTag.
- **Bot-message feedback:** relay pills and bot-bar send/receive pulses share one delivery handler. Read **`references/inter-bot-relay.md`** before changing lifecycle mapping, expiry, animation, or rail placement; never add a second event source or duplicate bot-color lookup.
- **Dense work surfaces (Repos pattern):** wireframe outline + spacing for separation, inset rounded pane cards, glass only on sticky chrome, active state = subtle background lift (never a border), clickable headers lift on hover — see `references/design-system.md` § Workspace surfaces.
