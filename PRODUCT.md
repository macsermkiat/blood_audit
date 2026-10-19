# blood_audit product scope

blood_audit audits completed adult inpatient red-cell and platelet orders at KCMH. It stores classifications with evidence and provenance for retrospective quality improvement. It does not authorize transfusion orders or provide bedside alerts.

The English and Thai documentation site serves operators, developers, and clinical reviewers. Its entry page should let each reader reach the right guide before presenting implementation details.

## Reader tasks

| Reader | Task | Documentation must provide |
| --- | --- | --- |
| Hospital operators | Prepare data, run the application, control access, generate reports | Working commands, prerequisites, expected outputs, and recovery limits |
| Developers | Maintain processing and verification | Current module contracts, source references, ordered rules, and required checks |
| Clinical reviewers | Interpret evidence and report denominators | Scope, classification meanings, policy precedence, and explicit validation limits |

## Content requirements

Describe current behavior in current guides. Keep ticket history and superseded instructions in dated records. Distinguish supported CLI operations from dataset-specific pilot scripts.

Operator examples must separate local execution from admitted model execution. A provider credential does not establish permission to submit clinical text. Explain the approved redactor, signed admission, database roles, and trusted authentication requirements beside the relevant task.

Clinical pages describe the implemented rules and their limits. Quote grounding and passing tests do not establish clinical truth. Explain routing exemptions and over-reservation findings separately from appropriateness judgments.

Use synthetic placeholders and synthetic screenshots. Clinical exports, encrypted data, partially replaced identifiers, and review outputs remain protected material. Never publish them as documentation examples.

## Language and layout

Use plain technical prose with exact command, field, and flag names. Put the task before implementation history. Remove repeated summary cards when they duplicate the introduction. Put cautions beside the action they govern.

English is the full reference. Thai pages cover the same routes, with some technical detail abbreviated and a link to the matching English page. Keep required setup and important limits consistent between languages. Noto Sans Thai and generous line height support Thai text.

Keep the existing warm neutral palette and red accent. Use color with text labels for scope and classification. Avoid promotional metrics, urgent bedside-alert styling, decorative emojis, and animated content that delays reading.

## Ownership

The site lives in `docs/`. [DESIGN.md](DESIGN.md) records visual decisions. [docs/CONTEXT.md](docs/CONTEXT.md) describes current module contracts. [docs/production-cli.md](docs/production-cli.md) describes the supported operational path. The [history index](docs/history/README.md) separates earlier build notes and plans from those references.
