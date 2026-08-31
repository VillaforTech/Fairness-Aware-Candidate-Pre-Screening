# Security policy

## Scope

The supported security surface is the current `fairness_project` package, CLI,
artifact bundle, and local v2 simulation API. Historical release tags remain
available for reproducibility but do not receive fixes.

The service is intended for localhost use with non-sensitive benchmark inputs.
It has no authentication, authorization, TLS termination, rate limiting,
multi-tenant isolation, or secrets-management layer.

## Report a vulnerability

Do not include exploit details, credentials, or sensitive data in a public
issue. Contact the maintainers privately through the repository owner's GitHub
profile. Include the affected revision, component, reproduction steps, impact,
and any suggested containment.

## High-priority findings

Please report issues that could:

- bypass bundle digest, schema, runtime, or policy validation;
- bypass quality-sidecar verification or substitute data-semantics evidence
  that does not match the model-ready CSV;
- load a governance-rejected artifact without an explicit override;
- make the API apply protected-attribute thresholds;
- expose request values in logs;
- accept extra fields or unseen categories silently;
- publish or consume a partial artifact bundle;
- alter or replace `monitoring.json` without detection, bypass its strict role
  schema, or cause a snapshot to contain source rows;
- execute code from an untrusted artifact beyond the documented `joblib`
  boundary; or
- disclose secrets committed to the repository or build output.

## Serialized-model warning

`model.joblib` uses Python object serialization. Load bundles only from a source
you trust. SHA-256 binding detects mutation relative to the manifest, but the
manifest is not signed. It cannot make a malicious bundle safe, because an
attacker can replace both the object and its recorded digest.

## Operational guidance

- Bind the service to `127.0.0.1`.
- Mount run bundles read-only when using containers.
- Do not expose a research override in a shared default environment.
- Do not submit personal, applicant, or confidential data.
- Rebuild artifacts after dependency or Python-minor changes.
- Review logs before sharing them, even though request values are not logged by
  the application.
- Treat `.quality.json` as evidence rather than a privacy boundary. It contains
  aggregate group counts and can expose small-cell information.
- Review monitoring snapshots before sharing them. They are aggregate-only, but
  rare category and protected-group counts can still reveal sensitive context.
- Keep raw or row-level monitoring inputs outside published run bundles.

This security policy does not authorize use of the project for employment or
other high-impact decisions.
