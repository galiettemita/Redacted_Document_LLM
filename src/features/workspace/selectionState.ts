export function getEffectivePassageId(
  hoveredPassageId: string | null,
  pinnedPassageId: string | null,
) {
  return hoveredPassageId ?? pinnedPassageId
}
