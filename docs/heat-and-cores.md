# Heat and cores

A thread that waits by checking a value in a loop ("spinning") keeps its core at 100 % while doing nothing. On a GB10 the
CPU and the GPU are one chip with one power budget and one cooler, so every watt a spinning core burns is heat the cooler
must also carry away for the GPU. Without the changes below, each Spark runs several such threads at 100 % (the table lists them by kind). With them,
no engine thread spins, at the same speed:

| What waited | How it waits | Spin removed | Speed against the same engine without it | Temperature (the Sparks' own sensors) |
|---|---|---|---|---|
| The engine's thread, for the GPU | asleep in the CUDA driver until the GPU signals (`TF_DS_CUDA_SCHED=blocking`) | 100 % of a fast core on three Sparks | replies equal; 0.3 % slower to 1.5 % faster | the spinning cores **10-14 °C cooler**, the chip's hottest zone 3-7 °C cooler (16 requests at once) |
| The first Spark's engine thread, for each GPU event | asleep until the event completes (`TF_DS_CUDA_EVENTS=blocking`) | 91 % -> 4 % of a fast core | replies equal; 0.5 % slower to 0.2 % faster | its fast cores 11.3 °C cooler |
| The engine's thread while it issues a prompt | at most 4 layers ahead of the GPU, then asleep (`TF_DS_ISSUE_LEAD=4`) | 100 % of a fast core on all four Sparks | replies equal; prompts unchanged | fast cores 3.8 °C cooler on average, 7.9 °C at their peak |
| The network threads, for the GPU's data (the decode rings' and [Rocinante](../ROCINANTE.md)'s, this repository's transport) | no thread in the transfer path: each transfer is a GPU-triggered send that the network card starts itself (`TF_DS_RING_WAIT`, `TF_DS_ROCI_WAIT`, on by default) | two to six threads a Spark at 100 %; at idle no engine thread on any Spark reaches 2 % of a core | outputs equal; speed level | |
| NCCL's exchange kernels and progress thread | not used: Rocinante carries every exchange | 2-4 of the GPU's 48 multiprocessors while waiting; a fast core at 100 % | prompts 8-29 % faster | 5 % less GPU energy per prompt (`nvidia-smi`, one run each) |
| Engram's reader (the model's Engram tables are read from disk), for finished disk reads | asleep between batches, at most 256 reads in flight (the drive's ceiling, ~293,000 reads a second) | its polling loop | replies equal; each submit 4.2 ms -> 0.07 ms | |

With the card's wait turned off (`TF_DS_ROCI_WAIT=0`, `TF_DS_RING_WAIT=0`), proxy threads post the transfers instead, and
they nap on the Arm CPU's timed wait-for-event instruction (WFET) between sends (`TF_DS_RDMA_WFE=1`, the default): 94-99 % of the time asleep, their small
cores 3-7 °C cooler than polling.

**Which core runs what.** Each GB10 has ten fast Cortex-X925 cores (5-9 and 15-19) and ten small Cortex-A725 cores (0-4
and 10-14). With the card waiting, the transport keeps no thread in the transfer path; what remains is a janitor thread that sleeps in
the kernel and looks for failures about ten times a second. When the card's wait is off, the proxy threads run on small
cores: the decode rings' on 0, 10 and 11 (`TF_DS_RDMA_CPUS`), Rocinante's first lane on 3 and 4 (`TF_DS_ROCI_CPUS`) and
its second on 1 and 2; the host lane's sleep in the kernel and are not pinned. The fast cores stay free for the engine's
own work.

**How it was measured.** Every test started from the same idle temperature, with all four Sparks read every 5 s under the
same load. Speed is the same benchmark as the [README's Performance section](../README.md#performance), against the same
engine without the change.
