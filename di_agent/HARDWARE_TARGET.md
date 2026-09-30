# Meter target: HW 4.2 Gen5 Riva

The user selected HW 4.2 Gen5 Riva as the target platform. This six-channel fault-location port targets **polyphase** meters. Singlephase V7/V8 streams do not supply the three current and three voltage channels required by the unchanged numerical pipeline.

The technical baseline is the supplied **Distributed Intelligence (DI) Software Development Kit (SDK) Documentation and Release Notes**, REV 007, July 22, 2026: chapter 2, “Meter HW 4.2 Gen5 Riva network technical specifications” (page 18), and the SDK 3.0.0 release compatibility table. The machine-readable counterpart is [hardware_profile.json](hardware_profile.json).

| Constraint | Documented HW 4.2 value | Current agent policy |
| --- | --- | --- |
| RAM | 65,000 KB | 2,048 KB |
| CPU | 30% (printed raw value 300000) | 2% |
| Flash | 40,000 KB | 2,048 KB |
| Maximum binary package | 10 MB | Check packaged artifact size |
| Container processes | 50 | Measure on target |
| Container open files | 1,250 | Measure on target |

The platform/container resources are shared capacity, not an automatic per-agent allocation. This release preserves the reference policy. Baseline host RSS of about 6 MB and the integration-suite peak of 24,800 KB exceed its 2,048 KB allocation. They are below the documented container RAM ceiling but **do not establish meter compliance**. Measure ARM/uClibc peak memory, CPU over the enforced policy interval, flash including dependencies/persistence, process count, and file descriptors with the other installed agents before qualification.

For SDK 3.0.0, the documented HW 4.2 firmware/AppServ pair is **50.10.312.2 / 2.0.581.0**. The older general architecture table's **10.2.512+ / 1.3.470** is not this port's compatibility baseline. The SDK package generator takes minimum versions; `info.sh` restricts `INSTALLON` to `GENX_PP=50.10.312.2,2.0.581.0`. Hardware IDs come from the SDK's GENX_PP mapping. These selectors restrict packaging and do not certify every later firmware pair. Confirm the exact installed firmware/AppServ combination before deployment.

## Future live acquisition

The document lists TC/two-cycle and HS/32K streams for HW 4.2 polyphase. A sub-second update interval alone does not establish that a stream supplies instantaneous waveform samples. The current pipeline must not consume the inherited one-second RMS values or treat TC aggregates as raw waveforms.

The installed `AgentApiHS.h` provides `ApiSubscribeToPeriodicDataHS`, `ApiGetConfigInfoHS`, and `ApiGetStatusInfoHS`. Its packed sample contains a sequence number, status, six signed 16-bit channels, and CRC. Header scale constants are **0.001 for current channels ch0/ch2/ch4** and **0.05 for voltage channels ch1/ch3/ch5**. The callback owns the samples only until it returns. The callback signature has no timestamp argument; the commented-out timestamp structure is not an available API.

Before implementing the adapter, establish the actual configured sample rate, A/B/C ordering, CT/PT scaling and primary/secondary units, timestamp/sequence mapping, status validity, CRC behavior, stream availability, and required policy permissions on the selected firmware. Copy callback data into bounded buffers, split gaps, then normalize to the core's `IA,IB,IC,UA,UB,UC` A/V contract. The HS label alone is not an exact rate or timing guarantee. Keep recording/segment offsets separate from SDK event time.

Live acquisition remains a later milestone. The unsigned ARM package includes the numerical core and output/configuration integration but no host replay parser or live subscription. It cannot claim fault detection on a physical meter until the adapter and HW 4.2 resource qualification pass. The supplied document establishes the chosen HW 4.2 capability; it does not provide enough HW 5.0 stream information to make a universal exclusivity claim.
