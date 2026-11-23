# Prediction Model Name Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the actual GPT model name beneath the Predictions heading only after a real provider prediction exists.

**Architecture:** `PredictionSidebar` will derive one display name from the existing real prediction metadata and render it in the heading block. The simulated demo and pre-prediction state remain unlabeled; CSS only adds the small secondary line requested in the screenshot.

**Tech Stack:** React 19, TypeScript, Testing Library, Vitest, CSS

**Spec:** `docs/superpowers/specs/2026-11-23-prediction-model-name-design.md`

## Global Constraints

- Use the model string already carried by a real prediction; do not hard-code a model name.
- Show no model name before a real prediction exists.
- Do not show the new model-name line for simulated predictions.
- Keep tokens, cost, retrieval, and implementation details hidden.

## Review Focus

- No predictions: the heading has no model-name line.
- Simulated-only predictions: the fictional demo does not imply a GPT model ran.
- Selected/active prediction: hiding that result from the card list does not hide its model name.
- Repeated results from the same model: the heading renders the name once.
- Partial prediction state: the model name and existing processing indicator coexist.

---

### Task 1: Render the real prediction model in the sidebar heading

**Files:**
- Modify: `src/features/predictions/PredictionSidebar.test.tsx`
- Modify: `src/features/predictions/PredictionSidebar.tsx`
- Modify: `src/styles/global.css`
- Modify: `CHECKLIST.md`

**Interfaces:**
- Consumes: `Prediction.model` and `Prediction.provenance` from `src/types.ts`.
- Produces: one `.prediction-model-name` line below the Predictions heading when `predictions.find(prediction => prediction.provenance === 'model')` exists.

- [x] **Step 1: Write the failing component tests**

Add tests asserting that a real active prediction displays `GPT-6 Astra` exactly once while the partial-state processing label remains, and that empty or simulated-only prediction arrays do not render the model-name line.

- [x] **Step 2: Run the focused test to verify RED**

Run: `npm test -- --run src/features/predictions/PredictionSidebar.test.tsx`

Expected: FAIL because the real model name is currently deliberately absent.

- [x] **Step 3: Implement the minimal sidebar and CSS change**

In `PredictionSidebar`, derive the first real prediction's `model`, wrap the heading and optional model line together, and leave `processing-label` as the other heading-row child. Style `.prediction-model-name` as muted secondary text beneath the heading.

- [x] **Step 4: Run focused and full verification**

Run: `npm test -- --run src/features/predictions/PredictionSidebar.test.tsx`

Expected: PASS.

Run: `npm test -- --run && npm run typecheck && npm run build`

Expected: 0 failed tests, 0 TypeScript errors, and a successful Vite production build.

- [ ] **Step 5: Record and ship the completed task**

Mark the model-name item complete in `CHECKLIST.md`, inspect the diff, commit the spec/plan/code/tests, and push `main`.
