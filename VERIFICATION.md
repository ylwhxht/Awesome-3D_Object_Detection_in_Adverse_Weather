# Verification — 12 October 2026

## Source integrity

- Parsed all 163 categorized entries into 151 records; repeated records retain their categories. Includes the README update of 12 October 2026.
- Preserved all 196 distinct paper, code, dataset, preprint, and project URLs from the source bibliography, with their original resource labels.
- All 10 source research tracks, entry publication metadata, and original update dates are retained.
- Source publication metadata is preserved, not independently re-audited.

## Executed checks

- `node --check app.js` and `node --check scene.js`: passed.
- `node scripts/build.mjs`: passed, static output generated.
- `node scripts/check.mjs`: passed. Checks source counts and links, identifiers, compound filters, normalized searches, sort order, saved-only selection, invalid URL parameters, HTML escaping, unique DOM IDs, and local file references.
- Local HTTP page: 200 response.
- Resource-filter regression checks passed: a project marked `code pending` does not count as available code, while `code&data` matches both resource filters.

## Browser verification

- Desktop overview and library visually inspected.
- Mobile at 390 × 844: navigation menu, research track selection, year selection, card layout and dark theme visually inspected.
- Narrow mobile at 320 × 740: no horizontal overflow after the header fix. Document width 305 px within a 320 px viewport including scrollbar.
- Combined search `v2x-r` + dataset track + 2025 + code found the expected single record.
- Reading-list save survived reload; test bookmark was removed after verification.
- Reading-list export produced `adverse-weather-reading-list.md`; downloaded title, venue, track and source URLs were inspected.
- Share URL included search and filter state; copy-to-clipboard returned the expected URL.
- Load more increased the displayed paper count from 8 to 16.
- An unmatched query displayed the empty state; Clear filters restored results.
- Sensor tabs responded to pointer and arrow-key navigation.
- Weather switching updated the illustration caption; play/pause state changed correctly.
- No browser error or warning logs were captured during the final checks.
- Temporary responsive viewport override was reset after testing.

Browser checks were performed in the Codex in-app browser. This was not a cross-engine Safari/Firefox test. External source links were preserved and checked for valid structure; their remote availability was not exhaustively rechecked.
