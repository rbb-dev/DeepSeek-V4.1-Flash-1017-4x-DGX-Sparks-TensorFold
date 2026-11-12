# Security

## Reporting a vulnerability

Report a security problem privately: open this repository's Security tab and choose "Report a vulnerability". Please do
not open a public issue for a problem that is not fixed yet. A problem in TensorFold's own engine or server, not
specific to this repository, should also be reported to [ashhart/TensorFold](https://github.com/ashhart/TensorFold)'s
Security tab.

Only the default branch's latest commit gets fixes.

## What is in scope

- The engine's HTTP API on the first Spark (port 8000, every address: `HOST=0.0.0.0`) and its keys: with an API key file,
  every route needs a key except `/health`, which without one says only that the engine is up, and `/metrics`, which is
  readable without a key by default (`METRICS_OPEN=1`; set it to 0 to require a key there too).
- The dashboard and its admin keys (`/admin`): off until the engine is given an admin key file; scoped keys; plain HTTP
  on the API's port.
- What `./setup.sh` installs as root on each Spark: the fabric rails' netplan file, the RoCE flow-control unit
  (`roce-pfc.service`) and, on a Spark without passwordless sudo, a sudo rule that lets the setup's user run one root
  helper (`/usr/local/sbin/dsv41-root`: the GPU clock limit and the page-cache drop) and nothing else.
- The RDMA copy between the Sparks (`tools/fabric-xfer`): a receiver accepts one connection, from the sender's addresses,
  after an HMAC challenge; it writes only beneath its destination folder.
- The files the engine keeps: the SSD tier's saved prompts, which hold conversation state, and the keys.

Out of scope: what the model writes, a model following instructions in text it was given, and a client sending more
work than four Sparks can do.

## Operating assumptions

The setup is made for four Sparks that serve this engine and nothing else. Each start drops the page caches and compacts
memory on all four, the setup changes their network configuration, and the API listens on every address. On a shared
network, keep port 8000 behind a firewall or a gateway that holds the keys, or reach the dashboard through an ssh tunnel.
The fabric copy does not encrypt the data it sends: keep the fabric private to the four Sparks.
