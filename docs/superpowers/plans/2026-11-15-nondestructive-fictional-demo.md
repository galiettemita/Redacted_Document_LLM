# Non-destructive Fictional Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the complete fictional interactive demo as an always-available, non-destructive presentation layer while preserving the real Andrew-method workflow and replacing customer-facing research diagnostics with a simple prediction UI.

**Architecture:** `App` keeps the real upload/workspace tree mounted and owns a modal demo flag. A dedicated `FictionalDemoWorkspace` consumes only restored sample assets and a deterministic mock prediction service; `DocumentWorkspace` remains HTTP/SSE-only. Real scientific metadata stays in backend state while the ordinary sidebar renders concise product states.

**Tech Stack:** React 19, TypeScript, Vitest, Testing Library, react-pdf, Vite, existing FastAPI backend unchanged.

**Spec:** `docs/superpowers/specs/2026-11-15-nondestructive-fictional-demo-design.md`

## Global Constraints

- Opening or closing the demo must never delete, replace, reset, or interrupt the real document.
- The demo is always labeled **Fictional interactive demo** and fake outputs are always labeled **Simulated prediction**.
- Real `DocumentWorkspace` output continues to come only from `createHttpPredictionService`; demo fixtures never cross that boundary.
- The customer-facing real UI must not display “visible-only context,” search query details, four-list retrieval, GPT-6 Astra, readiness gates, mask counts, query terms, or word-estimate metrics.
- Andrew's corpus, retrieval, prompt, evidence, and paid-call gates remain unchanged.
- No deployment or paid model call is authorized.

## Review Focus

- Opening the demo during an uploaded workspace must not invoke document deletion or unmount/reset the real workspace; Task 2 pins state preservation.
- Closing by button or Escape must restore focus to the opener and expose the real screen to assistive technology again; Task 2 pins focus and accessibility.
- Demo cancellation/reopening must not leak timers or duplicate progressive predictions; Task 1 pins cancellation and restart.
- A production result must never receive the simulated label and a demo result must never receive the model label; Tasks 1 and 3 pin both directions.
- Mobile document/prediction navigation, PDF hover/pin, zoom, and passage-card navigation must remain usable inside the modal demo; Task 1 pins shared interaction behavior and Task 4 runs browser QA when available.

---

### Task 1: Restore isolated fictional demo behavior

**Files:**
- Restore: `public/fictional-briefing.pdf`
- Create: `src/config.ts`
- Create: `src/data/sampleGeometry.json`
- Create: `src/data/sampleDocument.ts`
- Create: `src/services/mockPredictionService.ts`
- Create: `src/hooks/usePredictionSimulation.ts`
- Create: `src/features/demo/FictionalDemoWorkspace.tsx`
- Test: `src/services/mockPredictionService.test.ts`
- Test: `src/hooks/usePredictionSimulation.test.tsx`
- Test: `src/features/demo/FictionalDemoWorkspace.test.tsx`
- Modify: `src/types.ts`
- Modify: `src/features/predictions/PredictionCard.tsx`
- Modify: `src/features/predictions/SelectedPassage.tsx`

**Interfaces:**
- Produces: `SAMPLE_DOCUMENT`, `SAMPLE_PASSAGES`, and `SAMPLE_PREDICTIONS` for demo-only imports.
- Produces: `createMockPredictionService(options) -> PredictionService` and `usePredictionSimulation(document, service)`.
- Produces: `FictionalDemoWorkspace({ onClose, returnLabel, openerRef })` as the only product component allowed to import sample fixtures.
- Extends: `Prediction.provenance` to `'simulated' | 'model'`; cards render the exact corresponding label.

- [ ] **Step 1: Add failing service/hook tests for progressive fixture emission, abort cancellation, failure, empty completion, and clean restart.**

Run: `npm test -- --run src/services/mockPredictionService.test.ts src/hooks/usePredictionSimulation.test.tsx`
Expected: FAIL because the restored demo service and hook do not exist.

- [ ] **Step 2: Restore the deterministic mock service, simulation hook, fictional PDF/data, and provenance type until the focused tests pass.**

Run the Step 1 command; expected: PASS.

- [ ] **Step 3: Add a failing `FictionalDemoWorkspace` test proving the fictional label, simulated labels, progressive cards, hover/pin/card navigation, return control, Escape close, and zero backend API imports/calls.**

Run: `npm test -- --run src/features/demo/FictionalDemoWorkspace.test.tsx`
Expected: FAIL because the component does not exist.

- [ ] **Step 4: Implement the dedicated demo workspace using `PdfViewer`, `PredictionSidebar`, `workspaceReducer`, and the mock service only.**

Run the Step 3 command; expected: PASS.

- [ ] **Step 5: Commit the isolated demo foundation.**

Commit: `feat: restore isolated fictional demo`

### Task 2: Add the non-destructive presentation layer

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/App.test.tsx`
- Modify: `src/features/upload/UploadScreen.tsx`
- Modify: `src/features/upload/UploadScreen.test.tsx`
- Modify: `src/features/workspace/DocumentWorkspace.tsx`
- Modify: `src/features/workspace/DocumentWorkspace.test.tsx`
- Modify: `src/styles/global.css`

**Interfaces:**
- `UploadScreen` produces `onViewDemo()` from an always-visible **View fictional demo** button.
- `DocumentWorkspace` produces `onViewDemo()` from its header without changing document/service state.
- `App` owns `demoOpen: boolean`, keeps the real tree mounted, marks it `aria-hidden`/`inert` while the demo is open, and renders `FictionalDemoWorkspace` above it.

- [ ] **Step 1: Add failing App tests for demo entry from upload and uploaded workspace, preserved filename/workspace state, zero calls to `backendDocument.remove`, close/escape return, and focus restoration.**

Run: `npm test -- --run src/App.test.tsx`
Expected: FAIL because no demo entry or overlay exists.

- [ ] **Step 2: Add failing screen/workspace tests asserting the demo control is visible from both real screens and never replaces the current document.**

Run: `npm test -- --run src/features/upload/UploadScreen.test.tsx src/features/workspace/DocumentWorkspace.test.tsx`
Expected: FAIL on missing **View fictional demo** controls.

- [ ] **Step 3: Implement `demoOpen`, modal layering, inert/ARIA isolation, return-label selection, Escape handling, and focus restoration without unmounting the real tree.**

Run the Step 1 and Step 2 commands; expected: PASS.

- [ ] **Step 4: Add responsive demo-layer styles that reuse the existing workspace visual language and preserve the current layout.**

Run: `npm run typecheck && npm run build`
Expected: PASS.

- [ ] **Step 5: Commit the non-destructive entry flow.**

Commit: `feat: keep fictional demo always available`

### Task 3: Simplify the real prediction interface

**Files:**
- Modify: `src/features/workspace/DocumentWorkspace.tsx`
- Modify: `src/features/workspace/DocumentWorkspace.test.tsx`
- Modify: `src/styles/global.css`
- Modify: `README.md`
- Modify: `CHECKLIST.md`

**Interfaces:**
- Real sidebar user states are limited to **Preparing document…**, **Ready to predict**, **Predictions unavailable**, actionable error copy, the primary prediction button, and real prediction cards.
- Backend context/readiness records and Andrew-method scientific fields are unchanged and remain available to service/tests.

- [ ] **Step 1: Replace existing technical-copy assertions with failing tests for the concise states and explicit absence of every forbidden screenshot phrase/metric.**

Run: `npm test -- --run src/features/workspace/DocumentWorkspace.test.tsx`
Expected: FAIL because the research/debug panel still renders.

- [ ] **Step 2: Remove the visible-context metrics panel and map existing detection/context/analysis state to the concise product copy without changing scheduling behavior.**

Run the Step 1 command; expected: PASS.

- [ ] **Step 3: Remove obsolete panel CSS, document the permanent demo/real boundary, and check off the restored-demo milestone.**

Run: `rg -n "Visible-only context ready|Context query ready|four-list retrieval|visible query terms|confirmed masks|estimated words|visible words|readiness gate" src --glob '!**/*.test.*'`
Expected: no customer-facing matches.

- [ ] **Step 4: Commit the clean real-workspace UI.**

Commit: `refactor: simplify prediction readiness UI`

### Task 4: Regression verification and handoff

**Files:**
- Modify if required by evidence: tests and documentation only.

**Interfaces:**
- Consumes all prior tasks; produces a verified, pushed branch and a live loopback frontend.

- [ ] **Step 1: Run complete frontend verification.**

Run: `npm test -- --run && npm run typecheck && npm run build`
Expected: all tests PASS; build contains both explicitly labeled demo output and real provider path.

- [ ] **Step 2: Run complete backend regression verification.**

Run: `cd backend && uv run pytest`
Expected: all tests PASS except the documented optional host Tesseract skip.

- [ ] **Step 3: Scan the production import graph and bundle boundary.**

Run: `rg -n "sampleDocument|mockPredictionService|usePredictionSimulation" src --glob '!src/features/demo/**' --glob '!**/*.test.*'`
Expected: only `App` may import `FictionalDemoWorkspace`; no real workspace/service imports demo fixtures or mocks.

- [ ] **Step 4: Run browser QA if the in-app browser connector is available: upload a real PDF, open demo without losing it, exercise demo hover/pin/zoom/navigation/mobile view, close back to the same real document, and verify concise real copy.**

Expected: PASS, or document connector unavailability without substituting another browser mechanism.

- [ ] **Step 5: Update the checklist, verify the worktree, push all commits to `feat/real-retrieval-backend`, and report that no deployment or paid call occurred.**
