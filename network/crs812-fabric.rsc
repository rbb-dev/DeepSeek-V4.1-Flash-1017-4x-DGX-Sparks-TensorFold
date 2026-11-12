# MikroTik CRS812-8DS-2DQ-2DDQ (Marvell 98DX7335, RouterOS 7.25): the four-Spark RoCE v2 fabric with priority flow
# control, in the order it was built, from the factory state. See ../NETWORK.md for what each part does and how to check
# it. Before you start: take a restore point (/export file=before-fabric), and run it with no traffic on the fabric
# (changing L2MTU resets the switch chip; changing tx-manager queues pauses forwarding for a moment).
# The four Sparks' cables: two 400G QSFP-DD to 2x 200G QSFP56 breakout cables, one per QSFP-DD cage; the 200G lanes are
# -1 and -5 of each cage. Adjust the default-name lines to where your cables are.

# ---- management on ether1 alone; the fabric carries no management traffic
/interface list add comment="management: ether1 only" name=MGMT
/interface list member add interface=ether1 list=MGMT
/ip neighbor discovery-settings set discover-interface-list=MGMT
/tool mac-server set allowed-interface-list=MGMT
/tool mac-server mac-winbox set allowed-interface-list=MGMT
/ip service set ftp disabled=yes
/ip service set telnet disabled=yes
/ip service set www disabled=yes
/ip service set api disabled=yes
/ip service set api-ssl disabled=yes
/ip dhcp-client add interface=ether1 name=client1
# /user ssh-keys add user=admin key="<your ssh public key>"

# ---- the four Spark ports: 200G, jumbo frames, named after the Sparks (auto-negotiation on, advertising 200G only)
/interface ethernet set [ find default-name=qsfp56-dd-2-5 ] advertise=200G-baseCR4 l2mtu=9216 mtu=9000 name=node1
/interface ethernet set [ find default-name=qsfp56-dd-2-1 ] advertise=200G-baseCR4 l2mtu=9216 mtu=9000 name=node2
/interface ethernet set [ find default-name=qsfp56-dd-1-5 ] advertise=200G-baseCR4 l2mtu=9216 mtu=9000 name=node3
/interface ethernet set [ find default-name=qsfp56-dd-1-1 ] advertise=200G-baseCR4 l2mtu=9216 mtu=9000 name=node4

# ---- one hardware-offloaded bridge, VLAN 200 untagged on the four ports (the only bridge on the switch chip)
/interface bridge add comment="200G Spark fabric: hw-offloaded, VLAN 200" frame-types=admit-only-vlan-tagged mtu=9000 name=fabric vlan-filtering=yes
/interface bridge port add bridge=fabric comment="Spark fabric" edge=yes frame-types=admit-only-untagged-and-priority-tagged interface=node1 pvid=200
/interface bridge port add bridge=fabric comment="Spark fabric" edge=yes frame-types=admit-only-untagged-and-priority-tagged interface=node2 pvid=200
/interface bridge port add bridge=fabric comment="Spark fabric" edge=yes frame-types=admit-only-untagged-and-priority-tagged interface=node3 pvid=200
/interface bridge port add bridge=fabric comment="Spark fabric" edge=yes frame-types=admit-only-untagged-and-priority-tagged interface=node4 pvid=200

# ---- RoCE flow control: DSCP 26 and 0 into traffic class 3 with PFC; congestion notices (DSCP 48) strict priority
/interface ethernet switch qos profile add name=roce dscp=26 traffic-class=3
/interface ethernet switch qos profile add name=cnp dscp=48 traffic-class=6
/interface ethernet switch qos map ip add map=default dscp=0 profile=roce
/interface ethernet switch qos tx-manager queue set [find traffic-class=1] schedule=high-priority-group weight=1
/interface ethernet switch qos tx-manager queue set [find traffic-class=3] schedule=high-priority-group weight=1
/interface ethernet switch qos tx-manager queue set [find traffic-class=6] schedule=strict-priority
/interface ethernet switch qos priority-flow-control add name=pfc-tc3 traffic-class=3 rx=yes tx=yes
/interface ethernet switch qos settings set lossless-traffic-class=3
/interface ethernet switch qos port set node1 trust-l3=keep pfc=pfc-tc3 egress-rate-queue3=200Gbps
/interface ethernet switch qos port set node2 trust-l3=keep pfc=pfc-tc3 egress-rate-queue3=200Gbps
/interface ethernet switch qos port set node3 trust-l3=keep pfc=pfc-tc3 egress-rate-queue3=200Gbps
/interface ethernet switch qos port set node4 trust-l3=keep pfc=pfc-tc3 egress-rate-queue3=200Gbps
