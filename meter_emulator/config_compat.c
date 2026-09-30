/* Emulator-only bridge for the SDK 3 ARM API and supplied x86 DataServer.
 * ARM GetFeatureConfig has signature "ut" (agent, feature); the SDK host
 * DataServer accepts "t" (feature). Forward the real request and response.
 * No API result, metrology value, registration or policy result is mocked.
 * Never package this library with a production agent.
 */
#define _GNU_SOURCE
#include <dbus/dbus.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
static DBusMessage *adapt(DBusMessage *message) {
    DBusMessageIter iter;
    dbus_uint32_t agent;
    dbus_uint64_t feature;
    DBusMessage *copy;
    if (!dbus_message_is_method_call(message,
            "com.itron.owi.platform.dataserver", "GetFeatureConfig") ||
        strcmp(dbus_message_get_signature(message), "ut") != 0)
        return NULL;
    dbus_message_iter_init(message, &iter);
    dbus_message_iter_get_basic(&iter, &agent);
    if (agent != 0x23020000u && agent != 0x23020001u)
        return NULL;
    dbus_message_iter_next(&iter);
    dbus_message_iter_get_basic(&iter, &feature);
    copy = dbus_message_new_method_call(dbus_message_get_destination(message),
        dbus_message_get_path(message), dbus_message_get_interface(message),
        dbus_message_get_member(message));
    if (!copy) return NULL;
    if (!dbus_message_append_args(copy, DBUS_TYPE_UINT64, &feature, DBUS_TYPE_INVALID)) {
        dbus_message_unref(copy);
        return NULL;
    }
    fprintf(stderr, "EMU compatibility: GetFeatureConfig agent=%08x feature=%llx ut->t\n",
        agent, (unsigned long long)feature);
    return copy;
}

DBusMessage *dbus_connection_send_with_reply_and_block(DBusConnection *connection,
        DBusMessage *message, int timeout, DBusError *error) {
    typedef DBusMessage *(*Send)(DBusConnection *, DBusMessage *, int, DBusError *);
    Send send = (Send)dlsym(RTLD_NEXT, "dbus_connection_send_with_reply_and_block");
    DBusMessage *copy = adapt(message);
    DBusMessage *reply = send(connection, copy ? copy : message, timeout, error);
    if (copy) dbus_message_unref(copy);
    return reply;
}

dbus_bool_t dbus_connection_send_with_reply(DBusConnection *connection,
        DBusMessage *message, DBusPendingCall **pending, int timeout) {
    typedef dbus_bool_t (*Send)(DBusConnection *, DBusMessage *, DBusPendingCall **, int);
    Send send = (Send)dlsym(RTLD_NEXT, "dbus_connection_send_with_reply");
    DBusMessage *copy = adapt(message);
    dbus_bool_t result = send(connection, copy ? copy : message, pending, timeout);
    if (copy) dbus_message_unref(copy);
    return result;
}
