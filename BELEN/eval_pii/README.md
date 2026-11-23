# eval_pii

A bench to measure detectors of partial PII leaks, and a probe to ask BELEN what it knows about them, with controls that keep us from fooling ourselves.

---

## What each file does

| File | What it does |
|---|---|
| `leak_forge.py` | The **dose** axis. Synthetic personal data, partially redacted, with blanks, decoys and evasions, in English or Spanish. It also estimates LOD95, the smallest leak a detector catches 95% of the time. |
| `context_grammar.py` | The **context** axis. An audited factorial grammar with five conditions in two languages, where the same datum changes owner. |
| `image_forge.py` | The **image** axis. The same leaks rendered as screenshots, read back through OCR when Tesseract is available. |
| `reference_card.py` | Builds the reference card for any detector and holds the factorized probe (`FactorProbe`). |
| `detector_adapters.py` | Lets any NER, classifier or language model enter the reference card through a common interface. |
| `model_probe.py` | The bridge to BELEN: extracts internal states with the memory frozen, runs the random weight controls, writes the preregistration and gives the verdict. |
| `text_fingerprint.py` | **Exploratory.** What the model knows about a text from a short fragment: language, carrier and word order. No preregistration, no verdict. |
| `ladder.py` | The step by step runner for Colab. Each stage verifies before the next one spends GPU time. |

---

## Why a bench and not a benchmark

A partial leak is like a prescription with half the name covered by a thumb. A pharmacist still knows whose it is if the half that shows is the right half. The question is not whether a detector finds full names in clean text. The question is how little of a datum has to show before the detector notices, and whether it understands who the datum belongs to, or only reacts to the words around it.

So the bench has three axes, and each one asks one thing:

- **Dose**: how much of the datum is visible.
- **Context**: whose datum it is.
- **Image**: whether the answer survives leaving the text and entering a picture.

---

## The five context conditions

The datum is the same in every pair. Only its owner changes.

| Condition | Example (person against non person) | What it measures | Keyword baseline |
|---|---|---|---|
| seen | IP address of the customer / of the mail server | template memory | 1.00 |
| new owners | of the juror / of the hypervisor | knowing unseen words | about 0.5 |
| new frames | was logged as the customer ... | syntactic robustness, not understanding | 1.00 |
| contrast | of the customer, not of the firewall / the reverse | reading a negation | about 0.5 |
| incongruent | of whoever handles the mail server / of the unit that serves every customer | composing a head against its argument | 0.00 |

The last column is what a keyword baseline gets, measured. That is the certification: a condition only proves understanding if the keyword baseline fails on it. If words alone solve a condition, passing it says nothing about comprehension.

---

## How a verdict is reached

The model is tested against copies of itself that know nothing: the same architecture with random weights, several initializations. Like a blind tasting where the judge does not know which glass holds the wine that aged, and the cheap one is poured more than once.

A positive (the training adds something, it reads the negation, it composes) needs two things at once:

1. **An advantage over the strongest control.** The 95% interval of the paired difference against the best random model must clear 0.05.
2. **An absolute level.** The lower bound of the 95% interval must clear the bar: 0.65 for new owners, 0.60 for contrast and incongruent.

And a positive can still fall:

3. **Null rotation.** Each random control is evaluated as if it were the trained model, against the others. If a model that knows nothing gets the same positive, the trained one becomes not distinguishable from initialization noise.
4. **Resolution.** With K controls, the smallest p value you can claim is 1/(K+1). With the default 3, a positive is **provisional**. To claim it you need 19 controls (`python ladder.py prereg --controls 19`), and each one costs as much as the trained model.

The criteria and the predictions are written and sealed with hashes before measuring, in `PREREG.md`. The measurement refuses to run if the code changed afterwards, and that includes every file in `belen/`.

---

## How to run it

```bash
python ladder.py all
```

That runs `env, selftest, data, wiring, preflight, train, prereg` and stops. Before measuring:

1. Open `PREREG.md` in the work folder and write your predictions.
2. `python ladder.py ficha`
3. `python ladder.py pack`

With weights already trained:

```bash
python ladder.py all --ckpt run_raw.pt
```

To pass options to training, repeat `--env`:

```bash
python ladder.py all --env AXON_MODE=ach --env GEOMETRIC_DENDRITE=1
```

Each module checks itself with `python <file>.py --selftest`. The work folder can live in Drive so it survives a disconnection, and an interrupted card resumes where it stopped.

---

## Mistakes that showed up while building it

They stay in view because each one would have produced a result that looks good and means nothing.

| What happened | How it was caught | Fix |
|---|---|---|
| New sentences shared vocabulary, n-grams scored 1.00 | a result too good to be true | conditions certified by a keyword baseline |
| Gender in Spanish articles gave away the class | grammar audit with a word pair baseline | gender balanced in every list |
| People had one word, systems two: 0.97 with unseen owners | the full audit table | length balanced in every list |
| A shuffled probe flagged memorization on the cleanest signal | positive control | memorization measured as fit to noise in training |
| Random models read the negation (0.55) and composed (0.54) | a null run with the full configuration | absolute bars and advantage over the strongest control |
| Two random models differed by 0.11 with an interval that excluded zero | the same null run | several initializations of the control |
| The probe preferred a dead flow (constant states) | two controls at exactly 0.50 | choice per hypothesis, dead flows excluded |
| Evasive answers were read as no | adapter selftest | evasions count as unreadable |

---

## Limits

- Everything is synthetic. A card is valid for this matrix, this checkpoint and these seeds.
- The dose counts visible characters, not reidentification risk.
- The truth of each context is the rule of the grammar head. It is explicit and auditable, but it is a choice.
- Three data types appear in context (IP, phone, ID). Emails and names are personal by their form.
- The text fingerprint could also identify who wrote a text. Authorship is not probed here, on purpose.
- Log messages and help are in Spanish for now.
