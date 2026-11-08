# Redacted Document LLM Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished frontend-only React demo that locally renders PDFs and demonstrates simulated, linked predictions for redacted passages.

**Architecture:** A Vite React application owns document mode and selection state at the workspace boundary. Focused PDF, overlay, upload, and prediction components communicate through typed models, while deterministic mock data and timer-backed prediction behavior live outside presentation code.

**Tech Stack:** React 19, TypeScript, Vite, `react-pdf`/PDF.js, Vitest, Testing Library, CSS, and a static fictional PDF asset.

**Spec:** `docs/superpowers/specs/2026-11-07-redacted-document-demo-design.md`

## Global Constraints

- All document processing remains browser-local; there is no backend, persistence, authentication, external upload, OCR, retrieval, or real LLM integration.
- Display one configurable model name, exactly `Demo LLM`, with no selector, scores, confidence, or verification claims.
- Store all passage rectangles as normalized page-relative `x`, `y`, `width`, and `height` values in the 0–1 range.
- Preserve the original `Redaction_RAG_NAACL_v3.pdf` unchanged and outside the application bundle.
- Label the bundled document `Fictional demonstration document` and all generated predictions `Simulated prediction`.
- Clean up timers and object URLs whenever the document is replaced or the owning component unmounts.
- Support keyboard, visible focus, touch-equivalent selection, and responsive Document/Predictions switching.

## Review Focus

- A file named `.pdf` that PDF.js cannot parse must show an understandable unreadable-file error and remain local; Task 3 tests this validation path.
- Rapid document replacement during prediction streaming must not leak an old result into the new workspace; Task 4 tests cancellation.
- A passage with several rectangles must act as one hover/focus/pin target; Task 6 tests grouped activation.
- Drawing outside or across a rendered page edge must clamp to valid normalized coordinates; Task 5 tests geometry clamping.
- On narrow screens, activating a prediction must navigate and return the visitor to the Document view; Task 7 tests the responsive navigation state.

---

### Task 1: Application foundation and domain contracts

**Files:**
- Create: `package.json`, `package-lock.json`, `index.html`, `vite.config.ts`, `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json`
- Create: `src/main.tsx`, `src/App.tsx`, `src/config.ts`, `src/types.ts`, `src/test/setup.ts`
- Create: `src/styles/tokens.css`, `src/styles/global.css`
- Test: `src/App.test.tsx`

**Interfaces:**
- Produces: `PassageRect`, `Passage`, `Prediction`, `DocumentDescriptor`, `DocumentMode`, and `PredictionPhase` types; `MODEL_NAME = 'Demo LLM'`.

- [ ] **Step 1: Write the failing application-shell test** asserting the app name, heading `Explore what’s behind the redactions`, required support copy, `Choose PDF`, and `Demo — predictions are simulated` render with no sample action.
- [ ] **Step 2: Run `npm test -- --run src/App.test.tsx` and confirm it fails because the application is not scaffolded.**
- [ ] **Step 3: Create the Vite/TypeScript setup, domain types, design tokens, global document styling, and minimal `App`/`UploadScreen` shell that satisfies the test.**
- [ ] **Step 4: Run `npm test -- --run src/App.test.tsx` and `npm run typecheck`; expect both to pass.**
- [ ] **Step 5: Commit with `feat: scaffold document demo interface`.**

### Task 2: Fictional sample document and data

**Files:**
- Create: `scripts/generate-sample-pdf.mjs`
- Create: `public/fictional-briefing.pdf`
- Create: `src/data/sampleDocument.ts`
- Test: `src/data/sampleDocument.test.ts`

**Interfaces:**
- Consumes: domain types and `MODEL_NAME` from Task 1.
- Produces: `SAMPLE_DOCUMENT: DocumentDescriptor`, `SAMPLE_PASSAGES: Passage[]`, and `SAMPLE_PREDICTIONS: Prediction[]` with stable IDs and normalized geometry.

- [ ] **Step 1: Write a failing fixture-integrity test** asserting a multi-page fictional descriptor, ordered unique IDs, valid 0–1 rectangles, matching passage/prediction IDs, and at least one multi-rectangle passage.
- [ ] **Step 2: Run `npm test -- --run src/data/sampleDocument.test.ts`; expect missing fixture failure.**
- [ ] **Step 3: Implement the fictional briefing fixture and generate a matching multi-page PDF with black redaction blocks at the recorded normalized coordinates.**
- [ ] **Step 4: Run the fixture test and inspect PDF metadata/page count with the available local PDF utility; expect valid multi-page output and passing tests.**
- [ ] **Step 5: Commit with `feat: add fictional redacted briefing sample`.**

### Task 3: Local upload validation and document lifecycle

**Files:**
- Create: `src/features/upload/fileValidation.ts`
- Create: `src/features/upload/UploadScreen.tsx`
- Create: `src/hooks/useLocalPdf.ts`
- Modify: `src/App.tsx`
- Test: `src/features/upload/fileValidation.test.ts`, `src/features/upload/UploadScreen.test.tsx`, `src/hooks/useLocalPdf.test.ts`

**Interfaces:**
- Produces: `validatePdfFile(file: File): Promise<{ ok: true } | { ok: false; message: string }>`; `useLocalPdf()` with `openFile(file)`, `objectUrl`, `status`, `error`, and `clear()`.

- [ ] **Step 1: Write failing tests** for non-PDF rejection, corrupt PDF rejection, valid signature acceptance, drag state, understandable errors, and object URL revocation on clear/unmount.
- [ ] **Step 2: Run the three Task 3 test files; expect missing-module failures.**
- [ ] **Step 3: Implement PDF signature/parse validation, accessible drag-and-drop/file input behavior, local object URL ownership, and loading/error states.**
- [ ] **Step 4: Run Task 3 tests and `npm run typecheck`; expect all to pass.**
- [ ] **Step 5: Commit with `feat: add local PDF upload lifecycle`.**

### Task 4: Replaceable progressive prediction service

**Files:**
- Create: `src/services/predictionService.ts`
- Create: `src/services/mockPredictionService.ts`
- Create: `src/hooks/usePredictionSimulation.ts`
- Test: `src/hooks/usePredictionSimulation.test.tsx`

**Interfaces:**
- Produces: `PredictionService.stream(document, signal, onPrediction): Promise<void>` and `usePredictionSimulation(document, service)` returning `{ predictions, phase, error, reset }`.

- [ ] **Step 1: Write failing fake-timer tests** for pending → partial → complete order, deterministic empty/failure fixtures, unmount cleanup, and rapid-document-change cancellation without stale results.
- [ ] **Step 2: Run `npm test -- --run src/hooks/usePredictionSimulation.test.tsx`; expect missing-service failure.**
- [ ] **Step 3: Implement the AbortSignal-aware service contract, deterministic mock stream, and cleanup-owning hook.**
- [ ] **Step 4: Run the hook tests; expect all phases and cancellation assertions to pass.**
- [ ] **Step 5: Commit with `feat: simulate progressive passage predictions`.**

### Task 5: PDF viewer, controls, and marking geometry

**Files:**
- Create: `src/features/viewer/PdfToolbar.tsx`
- Create: `src/features/viewer/PdfViewer.tsx`
- Create: `src/features/viewer/drawingGeometry.ts`
- Create: `src/features/viewer/PassageOverlayLayer.tsx`
- Test: `src/features/viewer/drawingGeometry.test.ts`, `src/features/viewer/PdfToolbar.test.tsx`

**Interfaces:**
- Consumes: `DocumentDescriptor`, `Passage`, and normalized rectangles.
- Produces: page navigation/zoom/fit controls; `normalizeDrag(start, end, bounds): PassageRect`; viewer callbacks `onPageChange`, `onMarkPassage`, and `registerPassageTarget`.

- [ ] **Step 1: Write failing tests** for zoom/page bounds, fit action, drag-direction normalization, tiny-drag rejection, and clamping drags that start or end beyond page edges.
- [ ] **Step 2: Run the Task 5 tests; expect missing-module failures.**
- [ ] **Step 3: Configure the PDF.js worker, implement independently scrollable rendered pages, compact controls, responsive sizing, marking mode, and normalized rectangle creation.**
- [ ] **Step 4: Run Task 5 tests and `npm run typecheck`; expect all to pass.**
- [ ] **Step 5: Commit with `feat: add local PDF viewer and passage marking`.**

### Task 6: Linked passage overlays and prediction sidebar

**Files:**
- Create: `src/features/predictions/PredictionCard.tsx`
- Create: `src/features/predictions/SelectedPassage.tsx`
- Create: `src/features/predictions/PredictionSidebar.tsx`
- Create: `src/features/workspace/selectionState.ts`
- Modify: `src/features/viewer/PassageOverlayLayer.tsx`
- Test: `src/features/workspace/selectionState.test.ts`, `src/features/predictions/PredictionSidebar.test.tsx`, `src/features/viewer/PassageOverlayLayer.test.tsx`

**Interfaces:**
- Produces: `getEffectivePassageId(hoveredId, pinnedId)`; sidebar `onSelectPassage(id)`; overlay `hoveredId`, `pinnedId`, `onHover`, and `onPin` props.

- [ ] **Step 1: Write failing tests** for temporary-hover precedence and restoration, active-card omission without list reordering, grouped multi-rectangle activation, Enter/Space pinning, Escape/dismiss clearing, and no navigation on card hover.
- [ ] **Step 2: Run Task 6 tests; expect missing components/functions.**
- [ ] **Step 3: Implement cards, selected area, stable ordered list, grouped overlays, clamped compact preview, focus behavior, and muted-blue visual states while preserving black redactions.**
- [ ] **Step 4: Run Task 6 tests and accessibility assertions; expect all to pass.**
- [ ] **Step 5: Commit with `feat: link redactions with prediction cards`.**

### Task 7: Integrated workspace, uploaded-file demo, and responsive navigation

**Files:**
- Create: `src/features/workspace/DocumentWorkspace.tsx`
- Create: `src/features/workspace/workspaceReducer.ts`
- Modify: `src/App.tsx`
- Modify: `src/styles/global.css`
- Test: `src/features/workspace/DocumentWorkspace.test.tsx`, `src/App.test.tsx`

**Interfaces:**
- Consumes: upload lifecycle, viewer callbacks, sample fixtures, prediction service, and selection components.
- Produces: complete upload → workspace → replace flow, sample switching, one-mark uploaded demo, and desktop/mobile pane behavior.

- [ ] **Step 1: Write failing integration tests** for automatic sample processing, uploaded-file honesty copy, one marked sample prediction, `View interactive sample`, replacement reset, card-to-page pinning, and narrow-screen prediction activation returning to Document.
- [ ] **Step 2: Run Task 7 tests; expect integrated-workspace failures.**
- [ ] **Step 3: Implement the workspace reducer and orchestration, including timer/selection reset, independent pane scrolling, sample label, and Document/Predictions switch.**
- [ ] **Step 4: Run Task 7 tests, full `npm test -- --run`, and `npm run typecheck`; expect all to pass.**
- [ ] **Step 5: Commit with `feat: complete responsive document workspace`.**

### Task 8: Production polish and full verification

**Files:**
- Modify: `src/styles/global.css`
- Modify: `CHECKLIST.md`
- Create: `README.md`
- Test: all existing test files

**Interfaces:**
- Consumes: complete application.
- Produces: verified production build and documented local usage.

- [ ] **Step 1: Run `npm test -- --run`, `npm run typecheck`, and `npm run build`; fix any regressions before visual QA.**
- [ ] **Step 2: Read the local-preview skill reference, start the Vite preview, and browser-test upload errors, valid upload, rectangle marking, sample progressive cards, multi-line grouping, pin/dismiss, zoom/fit alignment, replacement, keyboard focus, and mobile switching.**
- [ ] **Step 3: Polish any observed layout, copy, focus, overflow, or alignment issues and rerun the affected tests plus the production build.**
- [ ] **Step 4: Mark every completed item in `CHECKLIST.md`, document setup and simulation limits in `README.md`, and inspect `git diff --check` plus `git status`.**
- [ ] **Step 5: Commit with `chore: verify and document frontend demo`, push all commits to `origin/main`, and report the exact verification results.**
