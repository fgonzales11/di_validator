"""Translate the bundled host DataServer's legacy subscription envelope.

Forward real DataServer samples to their registered ARM recipients as directed
SubscriptionUpdate method calls on the recipient's interface/object path.
LIDs and typed values remain unchanged; the ARM SDK decodes them.
This development bridge is never included in an agent package.
"""
import argparse
import json
import time
import os
from pathlib import Path

import dbus
import dbus.lowlevel
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib

INTERFACE = "com.itron.owi.platform.dataserver"
RECIPIENTS = {0x23020001: "com.itron.owi.platform.PVDetectionAgent"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    maximum = int(os.environ.get("DI_METER_METROLOGY_LIMIT", "0"))
    DBusGMainLoop(set_as_default=True)
    bus = dbus.bus.BusConnection("unix:path=" + str(args.socket))
    metrics = {"forwarded": 0, "transport": "broadcast signal -> directed method call",
               "signature": "aua(uv) -> a(uv)", "values": "unchanged",
               "started_utc": int(time.time())}
    args.metrics.write_text(json.dumps(metrics) + "\n")

    def forward(*values, **keywords):
        original = keywords["message"]
        if original.get_signature() != "aua(uv)":
            return
        # Only the actual DataServer owner may supply metrology.
        if original.get_sender() != bus.get_name_owner(INTERFACE):
            return
        recipients, readings = values
        for agent in recipients:
            if maximum and metrics["forwarded"] >= maximum:
                continue
            destination = RECIPIENTS.get(int(agent))
            if not destination:
                continue
            message = dbus.lowlevel.MethodCallMessage(
                destination, "/" + destination.replace(".", "/"),
                destination, "SubscriptionUpdate"
            )
            message.set_no_reply(True)
            rows = dbus.Array(readings, signature="(uv)")
            message.append(rows, signature="a(uv)")
            bus.send_message(message)
            metrics["forwarded"] += 1
            metrics["updated_utc"] = int(time.time())
            metrics["last_lids"] = [int(row[0]) for row in readings]
            temporary = args.metrics.with_suffix(".tmp")
            temporary.write_text(json.dumps(metrics) + "\n")
            temporary.replace(args.metrics)

    bus.add_signal_receiver(
        forward, signal_name="SubscriptionUpdate", dbus_interface=INTERFACE,
        message_keyword="message"
    )
    print("Metrology compatibility bridge ready", flush=True)
    GLib.MainLoop().run()


if __name__ == "__main__":
    main()
