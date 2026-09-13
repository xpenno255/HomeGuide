HomeGuide now refreshes the supported-manual catalogue in Home Assistant and lets
users create appliance entries, upload guides and confirm existing guides from
integration options. Manual searches are restricted to the selected appliance's
active, ready, confirmed documents.

For the reviewed Panasonic NN-ST46KB UK manual, mince-defrost questions return
Chaos Defrost programme 7 with entered weight and source references. The backend
checks the exact source file and 200–1200 g range. An optional HomeGuide Assist
conversation agent speaks these answers directly and delegates other requests
to your existing agent, avoiding an extra model generation for reviewed answers.

This is the first HACS release: add `xpenno255/HomeGuide` as a custom repository
of type Integration. Requires Home Assistant 2026.9 or newer. Restart Home
Assistant after installing. Configure the HomeGuide backend URL in the integration.

Update the backend image to `ghcr.io/xpenno255/homeguide:0.3.0` too; HACS updates
only the HA integration. Existing library data is preserved. Reviewed procedures
currently cover the exact UK microwave manual edition and self-contained English
mince-defrost questions. Other questions continue through ordinary retrieval.

See README.md and docs/reviewed-procedures.md for installation and Assist setup.
