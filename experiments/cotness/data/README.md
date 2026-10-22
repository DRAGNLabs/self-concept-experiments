# Neutral role-probe corpus

`neutral.jsonl` contains one paragraph from each of 144 distinct WikiText
articles, selected deterministically from the cached
[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext)
`wikitext-2-raw-v1` release, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.

WikiText is derived from Wikipedia articles and lists CC BY-SA 3.0 and GFDL
licenses. Each row retains the article title and source parquet filename/hash;
the paragraph text is unmodified apart from stripping surrounding whitespace.
The model-specific 192-token crop happens at runtime. Article titles identify
the underlying Wikipedia pages and their contributor histories.

The committed train/validation/test assignment uses different original dataset
splits and rejects repeated titles or text across splits. No deception examples
or model-generated reasoning are used to train the role probes. Reproduce with
`../scripts/prepare.py` from the repository root as documented in the experiment
README.
