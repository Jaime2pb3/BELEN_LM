# BELEN
**Boost Enhance Limits to Expose Negations**

A small language model that is not a transformer stack with a new coat of paint. It is a neuron drawn on paper, turned into code, and opened so other people can break it, fix it and grow it.

TwoQuarks Research · Luis Jaime Ledesma Pérez · ORCID 0009-0008-8796-0800

---

# How we work here
A result is only a result once it survives a check that could have failed. Everything else is a claim, a hypothesis, noise or perception.

Building a model is like learning to cook from someone who does not write recipes down. You can follow every step and still not know which one made the dish work, unless you change one thing at a time and taste after each change. So:
- **One change per pull request**, with a switch that turns it off.
- **A baseline with the same seed** next to every result.
- **A planted check**: a case where you know the right answer before you measure. If your test passes a build we know is broken, a pass on the real one says nothing.
- **Every result gets posted**, whichever way it points. A result that goes against the hypothesis is not a failure, it is the information that tells us where to go next.
- **Tests stay green.** `test_regression.py` protects the audited base. If a change has to move it on purpose, say so and update the reference in the same pull request.

Log messages and CLI help are still in Spanish. Translating them is a good first contribution.

---

## License

Apache License 2.0. You can use it, change it and build on it, including commercially. 
In return, keep the `LICENSE` and `NOTICE` files with any copy, mark the files you changed, and state where the work comes from. 
The license also grants the patent rights the contributors hold over their contributions.

If BELEN helps your work, cite it as:

Luis Jaime Ledesma Pérez. BELEN. 
TwoQuarks Research, 2026. https://github.com/Jaime2pb3/BELEN
