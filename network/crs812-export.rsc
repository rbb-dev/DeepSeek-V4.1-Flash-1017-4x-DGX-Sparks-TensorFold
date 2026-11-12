# /export terse of the switch these numbers were measured on (RouterOS 7.25, CRS812-8DS-2DQ-2DDQ),
# for reference: the serial number, software id, ssh key and time zone removed. RouterOS writes its export in its own
# order (the QoS ports before the lines that name them), so to build a switch from its factory state use
# crs812-fabric.rsc, which runs the same commands in an order that applies.
# by RouterOS 7.25
#
# model = CRS812-8DS-2DQ-2DDQ
/interface ethernet switch qos priority-flow-control add name=pfc-tc3 rx=yes traffic-class=3 tx=yes
/interface ethernet switch qos profile add dscp=26 name=roce traffic-class=3
/interface ethernet switch qos profile add dscp=48 name=cnp traffic-class=6
/interface ethernet switch qos map ip add dscp=0 profile=roce
/interface ethernet switch qos settings set lossless-traffic-class=3
/interface ethernet switch qos port set node4 egress-rate-queue3=200.0Gbps pfc=pfc-tc3 trust-l3=keep
/interface ethernet switch qos port set node3 egress-rate-queue3=200.0Gbps pfc=pfc-tc3 trust-l3=keep
/interface ethernet switch qos port set node2 egress-rate-queue3=200.0Gbps pfc=pfc-tc3 trust-l3=keep
/interface ethernet switch qos port set node1 egress-rate-queue3=200.0Gbps pfc=pfc-tc3 trust-l3=keep
/interface ethernet switch qos tx-manager queue set 1 schedule=high-priority-group weight=1
/interface ethernet switch qos tx-manager queue set 3 weight=1
/interface list add comment="management: ether1 only" name=MGMT
/ip neighbor discovery-settings set discover-interface-list=MGMT
/ip service set ftp disabled=yes
/ip service set telnet disabled=yes
/ip service set www disabled=yes
/ip service set api disabled=yes
/ip service set api-ssl disabled=yes
/interface bridge add comment="200G Spark fabric: hw-offloaded, VLAN 200; ether1 never a member" frame-types=admit-only-vlan-tagged mtu=9000 name=fabric vlan-filtering=yes
/interface ethernet set [ find default-name=qsfp56-dd-2-5 ] advertise=200G-baseCR4 l2mtu=9216 mtu=9000 name=node1
/interface ethernet set [ find default-name=qsfp56-dd-2-1 ] advertise=200G-baseCR4 l2mtu=9216 mtu=9000 name=node2
/interface ethernet set [ find default-name=qsfp56-dd-1-5 ] advertise=200G-baseCR4 l2mtu=9216 mtu=9000 name=node3
/interface ethernet set [ find default-name=qsfp56-dd-1-1 ] advertise=200G-baseCR4 l2mtu=9216 mtu=9000 name=node4
/interface ethernet switch set switch1 qos-hw-offloading=yes
/interface list member add interface=ether1 list=MGMT
/ip dhcp-client add interface=ether1 name=client1
/tool mac-server set allowed-interface-list=MGMT
/tool mac-server mac-winbox set allowed-interface-list=MGMT
# /user ssh-keys add user=admin key="<your ssh public key>"   (ours removed)
/interface bridge port add bridge=fabric comment="Spark fabric" edge=yes frame-types=admit-only-untagged-and-priority-tagged interface=node4 pvid=200
/interface bridge port add bridge=fabric comment="Spark fabric" edge=yes frame-types=admit-only-untagged-and-priority-tagged interface=node3 pvid=200
/interface bridge port add bridge=fabric comment="Spark fabric" edge=yes frame-types=admit-only-untagged-and-priority-tagged interface=node2 pvid=200
/interface bridge port add bridge=fabric comment="Spark fabric" edge=yes frame-types=admit-only-untagged-and-priority-tagged interface=node1 pvid=200
