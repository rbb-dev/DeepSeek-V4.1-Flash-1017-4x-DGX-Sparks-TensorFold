# Preparing the network: four DGX Sparks on one RoCE switch

How the four Sparks this repository is measured on are networked, so you can prepare your own RoCE switch the same way,
or set up the same MikroTik switch with this repository's configuration. The values below were read from the live
devices, and the behaviour described was measured on this hardware unless it is attributed to someone else.

**Result:** under a three-into-one overload at line rate (three Sparks sending to one at once), 0 drops and 0 RoCE resends (the switch's factory state: 1.45
million drops); under the real inference workload (tensor parallel over the four Sparks), 0 drops, 0 resends and 0
timeouts, at unchanged speed.

Files: [`network/crs812-fabric.rsc`](network/crs812-fabric.rsc) (the switch, from its factory state, in order),
[`network/crs812-export.rsc`](network/crs812-export.rsc) (the measured switch's whole configuration, for reference),
[`network/51-cx7-roce-static.yaml`](network/51-cx7-roce-static.yaml) (each Spark's addresses),
[`network/roce-pfc.service`](network/roce-pfc.service) and
[`network/install-roce-pfc.sh`](network/install-roce-pfc.sh) (each Spark's flow control, at every boot).
`./setup.sh network` applies each Spark's part for you (README, "Getting started"); the switch is yours to configure.

## 1. What the engine needs from the network

- **Two rails a Spark.** Each Spark has one 200G cable into its ConnectX-7's first QSFP port. The card reaches the GB10
  processor through two PCIe Gen5 x4 links, so that one port appears as two network devices ("rails"), each good for about
  100 Gb/s: rail A `enp1s0f0np0` (RDMA device `mlx5_0`) and rail B `enP2p1s0f0np0` (`mlx5_2`). The engine sends half of
  every transfer on each rail.
- **RoCE v2**, MTU 9000 end to end (the switch's L2 MTU 9216), one subnet per rail, every port forwarded in hardware.
- **Priority flow control for RoCE (recommended).** Without it, RoCE still runs, but it drops packets under overload
  and recovers by resending: measured 1-2 drops per million packets under inference, and 1.45 million drops in 20 s of a
  three-into-one overload (section 6). With priority flow control (PFC) on one traffic class: 0.

## 2. Your own switch: what to configure

Any switch that does the following works the same way (MikroTik's own commands are in section 3):

1. Four ports at 200G (or 400G ports split 2x 200G), each with one Spark; link up at 200 Gb/s with RS-FEC (the error correction 200G over copper
   needs; auto-negotiation selects it).
2. One layer-2 VLAN holding exactly the four ports, untagged, forwarded in hardware (not by the switch's CPU, which
   caps near 10 Gb/s); jumbo frames (L2 MTU 9216, IP MTU 9000).
3. Trust each packet's DSCP. Map **DSCP 0** (unmarked: the engine sends all its RoCE unmarked) **and DSCP 26** (the RoCE
   convention, for programs that mark their traffic) into one traffic class, here **3**; map congestion notifications
   (DSCP 48) into a strict-priority class, here 6.
4. **PFC on that class, both directions** (send and obey pause frames), with the class in the switch's no-drop buffer
   pool. Leave the switch's global 802.3x pause off (section 7).
5. Management on its own port, never in the fabric's VLAN.

The engine has no setting to mark its packets, so mapping DSCP 0 into the PFC class is what keeps its packets from being
dropped. Keep other traffic off the four ports' VLAN.

## 3. The MikroTik CRS812 measured here, from its factory state

| | |
|---|---|
| Switch | MikroTik CRS812-8DS-2DQ-2DDQ, chip Marvell 98DX7335, RouterOS 7.25 |
| Sparks | four GB10 (Dell Pro Max GB10), DGX OS 7.6, kernel 7.0.0-1019-nvidia, ConnectX-7 firmware 28.50.1002, rdma-core 2510.0.12 (NVIDIA's DOCA repository) |
| Cables | two 400G QSFP-DD to 2x 200G QSFP56 breakout cables (2 m, copper), one per QSFP-DD cage |

In this document the four Sparks are called node1 to node4: node1 is the first Spark (rank 0, the one that serves the
API); node2 to node4 are the workers.

| Spark | switch port (renamed) | breakout lane | rail A | rail B |
|---|---|---|---|---|
| node1 | node1 | qsfp56-dd-2-5 | 10.255.0.1/24 | 10.255.1.1/24 |
| node2 | node2 | qsfp56-dd-2-1 | 10.255.0.2/24 | 10.255.1.2/24 |
| node3 | node3 | qsfp56-dd-1-5 | 10.255.0.3/24 | 10.255.1.3/24 |
| node4 | node4 | qsfp56-dd-1-1 | 10.255.0.4/24 | 10.255.1.4/24 |

These are `scripts/config.sh`'s default addresses. Use any, as long as `RAIL_A_ADDRS` and `RAIL_B_ADDRS` in
`scripts/local.sh` match. One subnet per rail: every RoCE path has one route, and
no two interfaces of a host share a subnet (shared subnets cause ARP flux: RDMA packets leave the card and vanish).

Log in over ssh (`ssh admin@<switch>`), take a restore point (`/export file=before-fabric`), then run
[`network/crs812-fabric.rsc`](network/crs812-fabric.rsc)'s commands in order (paste them, or upload the file and run
`/import crs812-fabric.rsc`), with no traffic on the fabric. What the parts do:

- **Management:** `ether1` alone (interface list `MGMT`); ftp, telnet, www and the API off. `ether1` is never in the
  fabric's bridge, so a fabric link restart never cuts your ssh.
- **The four ports:** the QSFP-DD cages come up in 2x 200G breakout; the 200G lanes are `-1` and `-5` of each cage.
  Auto-negotiation stays on, advertising only 200G-baseCR4 (it negotiates RS-FEC). Changing the L2 MTU resets the switch
  chip, and every port blinks. Check: `/interface ethernet monitor node1,node2,node3,node4 once` gives `status: link-ok`
  and `rate: 200Gbps`.
- **One bridge, VLAN 200:** `/interface bridge port print` must show the `H` (hardware offload) flag on all four ports.
  Only one bridge per switch chip: a second bridge, or unused breakout lanes left in another bridge, stops the offload.
- **QoS for RoCE with PFC:** MikroTik's own RoCE example (the Quality of Service manual, RouterOS 7.17 and later), with
  ECN left off and one addition, DSCP 0 into class 3. Classes 1 and 3 share bandwidth equally, and class 6 is strict
  priority; `pfc-tc3` sends and obeys pause frames for class 3 (thresholds `auto`); `lossless-traffic-class=3` gives
  class 3 the no-drop buffer pool; the ports trust DSCP, and PFC's receive side needs the queue's egress rate set, even
  at wire speed. RouterOS 7.23 and later create the DSCP maps for profiles by themselves; DSCP 0 needs its own map line.
  Changing tx-manager queues pauses forwarding for a moment.

RouterOS keeps its configuration across reboots. [`network/crs812-export.rsc`](network/crs812-export.rsc) is the whole
configuration of the measured switch (`/export terse`, serial number and keys removed). RouterOS writes it in its own
order, so build from `crs812-fabric.rsc` and compare against the export.

## 4. Each Spark

**Addresses and MTU** (persistent, with netplan): [`network/51-cx7-roce-static.yaml`](network/51-cx7-roce-static.yaml)
as `/etc/netplan/51-cx7-roce-static.yaml` (mode 600), with N set to the last part of this Spark's addresses (1 to 4 with
the defaults, 33 to 36 in the table above); then `sudo netplan generate && sudo netplan apply`. NetworkManager on a
desktop install deletes fabric addresses added by hand, so use netplan. The DGX OS defaults protect against ARP flux
(`arp_ignore=1`, `arp_announce=2`, `rp_filter=2`). Check a jumbo ping that may not fragment, between every pair on both
rails: `ping -M do -s 8972 10.255.0.<peer>` and `ping -M do -s 8972 10.255.1.<peer>`.

**RDMA:** with NVIDIA's rdma-core the devices are `mlx5_0` and `mlx5_2` (Ubuntu's rdma-core names them `rocep1s0f0` and
`roceP2p1s0f0`). `rdma link show`: both `ACTIVE`, `LINK_UP`. The engine finds each rail's device from `RAIL_A_IF` and
`RAIL_B_IF`, and uses each device's first RoCE v2 GID that holds an IPv4 address (3 on the Sparks measured here; check
with `cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/3`, which gives `RoCE v2`); `TF_DS_RDMA_GID` names another.
Both PCIe functions should be at x4 (`/sys/bus/pci/devices/<bdf>/current_link_width`); after a cold boot a function can
come up at x2: check the width, and reboot warm if it does.

**Flow control on the card** (DSCP 0 and 26 to priority 3, PFC on priority 3), on both rails:
```
sudo dcb app add dev enp1s0f0np0 dscp-prio 0:3 26:3
sudo dcb app add dev enP2p1s0f0np0 dscp-prio 0:3 26:3
sudo dcb pfc set dev enp1s0f0np0 prio-pfc 3:on
sudo dcb pfc set dev enP2p1s0f0np0 prio-pfc 3:on
```
The first DSCP entry switches the card to DSCP trust (`mlnx_qos -i enp1s0f0np0`: `Priority trust state: dscp`,
`PFC configuration: enabled 0 0 0 1 0 0 0 0`). Enabling PFC turns the card's 802.3x pause off (`ethtool -a`: RX off, TX
off), and setting PFC can restart the link for a few seconds. Check: `dcb pfc show dev <rail>` gives `3:on`, and
`dcb app show dev <rail>` gives `dscp-prio 0:3 AF31:3` (AF31 is DSCP 26).

**These settings do not survive a reboot or a driver reload.** [`network/roce-pfc.service`](network/roce-pfc.service)
puts them back at every boot (`dcb app replace`: safe to run again). Install it on every Spark with
`network/install-roce-pfc.sh --start user@spark1 user@spark2 user@spark3 user@spark4` (it needs passwordless sudo),
or by hand: `sudo install -m 644 roce-pfc.service /etc/systemd/system/ && sudo systemctl daemon-reload &&
sudo systemctl enable --now roce-pfc.service`. After a driver reload, run `sudo systemctl restart
roce-pfc.service`. Tested here by rebooting each Spark: the unit ran at boot, the settings came back, and the
overload test gave 0 drops.

## 5. The engine

- With DSCP 0 in the PFC class on both ends, the engine's RoCE drops nothing, with no setting of its own.
- The engine's container needs `--device /dev/infiniband`, `--ulimit memlock=-1` and `--cap-add IPC_LOCK`; `./start.sh`
  passes them.
- The engine does not use NCCL. If you run NCCL yourself (for the check in section 6), name both rails:
  `NCCL_IB_HCA=mlx5_0,mlx5_2`, with `NCCL_SOCKET_IFNAME=enp1s0f0np0`, `NCCL_CROSS_NIC=1` and
  `NCCL_IB_SUBNET_AWARE_ROUTING=1`.

## 6. Checking it

| check | how | expected |
|---|---|---|
| switch links | `/interface ethernet monitor node1,node2,node3,node4 once` | link-ok, 200Gbps |
| full rate, no congestion | perftest on each rail, each Spark writing to the next: on the receiver `ib_write_bw -d mlx5_0 -x 3 -s 1048576 --report_gbits -D 10 -F`, on the sender the same plus the receiver's rail A address (and `-d mlx5_2` with the rail B address) | ~98 Gb/s a flow, 0 pause frames |
| overload | three Sparks write into one at once on both rails (six flows of the command above, each on its own port `-p`, 20 s, the engine stopped) | switch drops 0; the Sparks' `out_of_sequence` and `packet_seq_err` unchanged; ~196 Gb/s into the one; the switch pausing the senders: their `rx_prio3_pause` rising (~4.25 million each in 20 s here) |
| NCCL (optional) | nccl-tests `all_reduce_perf` over the four, both rails | ~24 GB/s bus bandwidth at 16 GB (measured here) |
| the real workload | the counters below, before and after a stretch of serving | 0 drops, 0 resends, 0 timeouts |

The counters: on the switch, `/interface ethernet print stats` (tx-drop-packet, tx-drop-queueN-packet, tx-pause,
rx-pause; RouterOS prints them transposed, one column per port, with spaces as thousands separators) and
`/interface ethernet switch qos port print pfc` (per class); on each Spark,
`/sys/class/infiniband/mlx5_{0,2}/ports/1/hw_counters/` (`out_of_sequence`, `packet_seq_err`, `local_ack_timeout_err`)
and `ethtool -S enp1s0f0np0` (`rx_prio3_pause`, `tx_prio3_pause`, `rx_discards_phy`).

**Measured here** (the overload test above, 20 s, and the engine serving):

| state | overload: switch drops | RoCE resends | into the one Spark | serving: drops / resends |
|---|---|---|---|---|
| factory (no flow control, no QoS) | 1,454,393 | 136,495 | 159.8 Gb/s | ~1-2 drops per million packets |
| 802.3x flow control on | 2,275,830 | 137,086 | 160 Gb/s | - |
| ECN on the default queue | 710,206 | 5,039 | 196 Gb/s | 1,340 drops in 15 min; prompts 3 % slower |
| **PFC on class 3 (this document)** | **0** | **0** | **195.9 Gb/s** | **0 / 0 over 32 min (~714 million packets a port); speed unchanged** |

## 7. What does not work on this switch, and why

- **802.3x flow control** (`rx-flow-control` / `tx-flow-control`). With QoS hardware offloading on, which this switch
  generation forces, the switch never sends pause frames toward senders; it only obeys a receiver's pause, which makes
  drops worse (2.28 million against 1.45 million). MikroTik's forum (thread 273446) reports the same on a CRS310.
  Changing the setting also renegotiates each link (8-20 s down).
- **ECN marking on the default queue alone.** The ConnectX-7 marks RoCE as ECN-capable and runs DCQCN by default, so
  marking does slow senders: it cut overload drops by about half (and RoCE resends by more than 95 %), but under the real workload it cost 3 % of speed and
  did not stop drops.

## 8. Rolling it back

- **Switch:** `/interface ethernet switch qos port set node1,node2,node3,node4 trust-l3=ignore pfc=disabled`,
  `/interface ethernet switch qos map ip remove [find dscp=0]`, `/interface ethernet switch qos settings set
  lossless-traffic-class=auto`; then remove the profiles `roce` and `cnp` and the PFC profile `pfc-tc3`, reset `egress-rate-queue3` on the four
  ports and the tx-manager queue schedules of classes 1, 3 and 6. Or import your restore point, which undoes all of it.
- **Sparks:** `sudo systemctl disable --now roce-pfc.service`, then on each rail `sudo dcb pfc set dev <rail>
  prio-pfc 3:off` and `sudo dcb app del dev <rail> dscp-prio 0:3 26:3`; or disable the unit and reboot.
