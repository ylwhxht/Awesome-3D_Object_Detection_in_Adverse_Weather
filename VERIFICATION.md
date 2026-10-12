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
- Runtime asset references include content-based cache versions; validation checks each version against the current file contents.

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

## Typography, section order and motion refinement

- Enlarged the body, navigation, card and control text; verified desktop, 390 × 844 and 320 × 740 layouts. Neither mobile width had horizontal overflow.
- Main content now follows Research Map → Field Notes → community contribution → The Reading Room. Desktop and mobile navigation follow the same research / notes / library order.
- Research-map selection still scrolls to the library and applies its filter: Weather simulation returned 21 resources. The Radar field-note action returned 11 resources in 4D radar & camera.
- Replaced the nearly static scene with moving point clouds, scan planes and rings, radar rotation, tracked volumes, weather particles and pointer parallax. Labels were adjusted after inspecting small screens.
- Checked all four weather controls and captions. Explicit play and pause choices survive reload; the initial reduced-motion setting still starts paused.
- Two captures of the paused canvas were identical. Running captures showed changed scan positions and geometry. No warning or error logs appeared during these checks.

Browser checks were performed in the Codex in-app browser. This was not a cross-engine Safari/Firefox test. External source links were preserved and checked for valid structure; their remote availability was not exhaustively rechecked.
