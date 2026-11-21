# DriftGuard Documentation

Engineering documentation for DriftGuard. Each document states whether its
claims are **measured**, **inspected** or **assumed**, and documents that
describe planned rather than implemented behaviour say so in their title.

## Status

| Area | Document | Status |
|---|---|---|
| **Audit** | [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md) | ✅ Complete — baseline at `fb8c56b` |
| **Roadmap** | [`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md) | ✅ Complete — requirements, phases, traceability |
| **Decisions** | [`adr/`](adr/README.md) | 🟡 0001–0004 retrospective, 0005 accepted; further ADRs scheduled |
| **Modelling** | `modelling/` | ⬜ Scheduled — roadmap phase P3 |
| **Architecture** | `architecture/` | ⬜ Scheduled — phase P4 |
| **Testing and QA** | `quality/` | ⬜ Scheduled — phase P5 |
| **Security and deployment** | `security/`, `deployment/` | ⬜ Scheduled — phase P5 |
| **UX and usability** | `ux/` | ⬜ Scheduled — phase P6 (inspection findings currently in audit §4) |
| **Business strategy** | `business/` | ⬜ Scheduled — phase P7, blocked on research not yet performed |

Empty sections are listed deliberately. A directory of placeholder files would
imply work that has not happened.

## Where to start

- **Assessing the codebase?** → [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md)
- **Planning work?** → [`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md)
- **Wondering why something is the way it is?** → [`adr/`](adr/README.md)
- **Using the application?** → [`../README.md`](../README.md)
- **Presenting it?** → [`../PRESENTATION_GUIDE.md`](../PRESENTATION_GUIDE.md)

## Evidence convention

| Mark | Meaning |
|---|---|
| **[M]** | Measured by running a command; the command is shown |
| **[I]** | Inspected in source, with `file:line` |
| **[A]** | Assumption, estimate or judgement — not evidence |

This exists because DriftGuard's product claim is auditability. Documentation
that blurred measurement and opinion would undercut the thing the tool is for.
