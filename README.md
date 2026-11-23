# Hungarian Tax Information Assistant


Imagine a mid-sized Hungarian company. Tax questions come up in many departments every week:
- Finance needs to know when the corporate tax return is due.
- Controlling asks which items reduce the tax base.
- Management asks which tax allowances the company can claim.

The answers are in the official information booklets of the Hungarian tax authority (NAV), but these are long PDF documents. So people either search through them by hand, or send the question to the tax team, which answers the same questions again and again.

This app lets any employee ask such a question in plain language and get a short answer, with a reference to the exact booklet and section it came from.

## What the app does

The app answers questions about Hungarian corporate income tax (*társasági adó*) using RAG (Retrieval-Augmented Generation):
1. It searches the booklets for the passages most relevant to the question.
2. A language model writes a short answer using only those passages.
3. The answer is returned together with its sources, so the user can check it.

If the booklets do not cover the question, the app says so instead of guessing. Questions can be asked in Hungarian or English.

## Example questions

- *"Mikor kell benyújtani a társasági adó bevallását?"* ("When is the corporate tax return due?")
- *"Milyen tételek csökkentik a társasági adó alapját?"* ("Which items reduce the corporate tax base?")
- *"Ki a társasági adó alanya?"* ("Who is subject to corporate income tax?")
- *"Mi a legjobb pizza feltét?"* ("What is the best pizza topping?"): not covered, so the app says it cannot help.

## Data

- The 4 NAV information booklets on corporate income tax (*Információs füzetek*, 2026 edition): official, public, Hungarian-language PDFs.
- This is a deliberately small example set. Other tax types, such as VAT or payroll taxes, can be added by adding their booklets. The application itself does not change.

## Limitations

- It answers lookup questions: rules, deadlines, obligations and conditions written in the booklets. It does not calculate taxes or analyse company data.
- Its knowledge is limited to the selected 2026 booklets.
- It provides information, not tax advice.
