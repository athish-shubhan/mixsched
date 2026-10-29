# Traces

`trace_seren.csv` and `trace_kalos.csv` are the job traces of the Acme
release (Hu et al., "Characterization of Large Language Model Development in
the Datacenter", NSDI '24), https://github.com/InternLM/AcmeTrace, licensed
CC-BY-4.0. They are not committed here; fetch them with:

```bash
git clone --depth 1 https://github.com/InternLM/AcmeTrace /tmp/AcmeTrace
cp /tmp/AcmeTrace/data/job_trace/trace_*.csv data/
```
