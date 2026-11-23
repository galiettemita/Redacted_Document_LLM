# Prediction model name design

## Intent

Show which GPT model produced the real results without adding technical clutter or implying that a model has run before a prediction exists.

## Behavior

- After the first real provider prediction arrives, show its existing display model name directly beneath the **Predictions** heading.
- Before any real prediction exists, show no model name.
- Keep the fictional demonstration unchanged; simulated output must not present a GPT model name in this new location.
- Continue hiding token, cost, retrieval, and implementation metadata.

## Approaches considered

1. **Use the model name carried by the real prediction (selected).** This is truthful, requires no hard-coded product label, and automatically follows a future backend model change.
2. **Always show a hard-coded configured model.** This would fill the space before prediction, but could become inaccurate and conflicts with the requested after-prediction timing.
3. **Add model identity to backend readiness.** This could show the future model before prediction, but expands the API without helping the selected behavior.

## Implementation boundary

The existing prediction sidebar will derive the label only from a prediction whose provenance is `model`, render it under the section heading, and receive a focused component regression test. No backend, prediction, demo, or card behavior changes.
