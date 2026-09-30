"""Apply the maintained port sources to the separate SDK clone (run in WSL)."""

from pathlib import Path
import argparse
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import uuid
import json


def replace_function(text, name, body):
    match = re.search(r"(?:static\s+)?void\s+" + re.escape(name) + r"\([^;{}]*\)[^;{}]*\{", text)
    if not match:
        raise ValueError("Function not found: " + name)
    start = match.end()
    depth = 1
    end = start
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[:start] + "\n" + body + "\n}" + text[end:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdk", type=Path, default=Path("/root/DI-SDK-3.0.0/DistributedIntelligenceSDK"))
    args = p.parse_args()
    sdk = args.sdk.resolve()
    project = sdk / "Agents/FaultLocationAgent"
    original = sdk / "Agents/ThirdPartyReferenceAgent"
    source = Path(__file__).resolve().parents[1]
    assert project.parent == sdk / "Agents" and project.is_dir()
    # Reapply from the reference template for reproducibility; never edit the reference.
    for name in ["agent_config.cpp", "agent_daemon.cpp"]:
        content = (
            (original / "src" / name)
            .read_text()
            .replace("ThirdPartyReferenceAgent", "FaultLocationAgent")
            .replace("Third Party Reference Agent", "Fault Location Agent")
        )
        (project / "src" / name).write_text(content)
    for name in ["agent_types.h", "configs/agent_config.h"]:
        shutil.copy2(original / "include" / name, project / "include" / name)
    for name in ["fault_core.hpp", "fault_wire.hpp"]:
        shutil.copy2(source / "core" / name, project / "include" / name)
    for name in ["fault_core.cpp", "fault_wire.cpp"]:
        shutil.copy2(source / "core" / name, project / "src" / name)
    for name in ["replay_io.hpp"]:
        shutil.copy2(source / "host" / name, project / "include" / name)
    shutil.copy2(source / "host/replay_io.cpp", project / "src/replay_io.cpp")
    for name in ["agent_feature.cpp", "agent_metrology.cpp"]:
        shutil.copy2(source / "integration" / name, project / "src" / name)
    shutil.copy2(source / "integration/agent_feature.h", project / "include/agent_feature.h")
    types = project / "include/agent_types.h"
    content = types.read_text().replace(
        "#include <diagent_log.h>", "#include <diagent_log.h>\n#include <map>\n#include <string>"
    )
    content = re.sub(
        r"struct stAgentFeatureConfig\s*\{.*?\};",
        "struct stAgentFeatureConfig { std::map<std::string,std::string> parameters; };",
        content,
        flags=re.S,
    )
    types.write_text(content)
    config = project / "src/agent_config.cpp"
    text = config.read_text()
    text = re.sub(
        r"static stAgentFeatureConfig agentFeatureConfig=.*?;",
        "static stAgentFeatureConfig agentFeatureConfig;",
        text,
    )
    text = replace_function(
        text,
        "UpdateAgentFeatureConfigurations",
        "    GetAgentFeature().UpdateAgentFeatureConfig(i_GenAgentFeatureConfig, i_AgentFeatureConfig);",
    )
    text = replace_function(
        text,
        "ApplyConfig",
        """    (void)i_eLogLevel;
    stGeneralConfig general;
    stAgentFeatureConfig feature;
    stCoutLogConfig coutConfig;
    stCallBackConfig callbackConfig;
    stConfiguration configuration;
    stArrayBufferConfig arrayConfig;
    pthread_mutex_lock(&g_ConfigCBmutex);
    general=genAgentFeatureConfig; feature=agentFeatureConfig;
    coutConfig.uiIsEnabled=s_coutIsEnabled; coutConfig.uiNumLogFiles=s_coutNumLogFiles; coutConfig.uiFileSize=s_coutFileSize;
    callbackConfig=stCllbckConfig; configuration=stCfg; arrayConfig=stArrayBuffConfig;
    pthread_mutex_unlock(&g_ConfigCBmutex);
    if(i_featureId==kuiAgentFeatureID) UpdateAgentFeatureConfigurations(general,feature);
    else if(i_featureId==kuiGenericFeatureID) UpdateGenFeatureConfgurations(coutConfig,callbackConfig,configuration,arrayConfig);""",
    )
    text = replace_function(
        text,
        "SetAgentFeatureThresholdVoltageCB",
        "    (void)i_folderName; (void)i_paramName; (void)i_stConfigVal;",
    )
    text = replace_function(
        text,
        "SetAgentFeatureCheckForVoltageCB",
        "    (void)i_folderName; (void)i_paramName; (void)i_stConfigVal;",
    )
    # Exact decimal strings avoid ConfigParser's float32 conversion.
    insertion = """
static void SetFaultParameter(const string&,const string& name,const stConfigVal& value) {
    if(value.type!=common::STRING || !value.unParamVal.sVal) return;
    const std::string copy(value.unParamVal.sVal);
    pthread_mutex_lock(&g_ConfigCBmutex);
    agentFeatureConfig.parameters[name]=copy;
    pthread_mutex_unlock(&g_ConfigCBmutex);
}
"""
    location = text.index(" static void RegisterAgentFeatureConfigCallbacks()\n")
    text = text[:location] + insertion + text[location:]
    text = replace_function(
        text,
        "RegisterAgentFeatureConfigCallbacks",
        """    common::CConfigParser& configParser=common::CConfigParser::GetConfigParserInstance();
    string error;
    configParser.RegisterConfigParamCallback(kuiAgentFeatureID,folder_general,FeatureState,SetAgentFeatureGenFeatureEnabledCB,error);
    configParser.RegisterConfigParamCallback(kuiAgentFeatureID,folder_general,LogLevel,SetAgentFeatureGenLogLevelCB,error);
    configParser.RegisterConfigParamCallback(kuiAgentFeatureID,folder_general,KeepLogLevelForSecs,SetAgentFeatureGenKeepLogLevelForSecsCB,error);
    const std::map<std::string,std::string> defaults=fault::Config().values();
    for(std::map<std::string,std::string>::const_iterator it=defaults.begin();it!=defaults.end();++it)
        configParser.RegisterConfigParamCallback(kuiAgentFeatureID,"FaultLocation",it->first,SetFaultParameter,error);""",
    )
    text += """
#ifdef NODAEMON
bool ApplyFaultHostConfiguration(const std::string& xml) {
    static uint32_t version=1;
    common::XmlReturnVal result=common::eXML_SUCCESS;
    const bool ok=common::CConfigParser::GetConfigParserInstance().ManageConfigXMLString(
        kuiAgentFeatureID,version++,0,static_cast<uint32_t>(xml.size()),xml.c_str(),result);
    if(ok) ApplyConfig(kuiAgentFeatureID);
    return ok;
}
#endif
"""
    config.write_text(text)
    daemon = project / "src/agent_daemon.cpp"
    text = daemon.read_text()
    text = "#include <agent_feature.h>\n" + text
    # Normal termination must retain outcomes and execute SDK atexit hooks.
    text = text.replace("Finalize();\n\t\t\t_exit(1);", "Finalize();\n\t\t\treturn;")
    text = text.replace(
        "ProcessSigusr1Command();",
        "ProcessSigusr1Command();\n            if(!g_stFeaturesStatus.bIsAgentFeatureStarted) GetAgentFeature().StopReplay();",
    )
    text = replace_function(
        text,
        "Finalize",
        """    GetAgentFeature().StopReplay();
    finalize();
    g_abPresentStatus[AGENTFEATURE]=false;
    g_abPresentStatus[COMMANDEXECUTION]=false;
    g_abPresentStatus[GENERICFEATURE]=false;
    g_abPresentStatus[CONFIGFEATURE]=false;
    CCommonMetrology::GetCommonMetrologyInstance()->ReleaseCommonMetrologyInstance();
    resourcelogger::stopResourceLogger();
    // ApiDeregisterAgent belongs to uninstall; retaining registration preserves outcomes.
""",
    )
    text = text.replace("signal(SIGKILL, SignalHandler);", "// SIGKILL cannot be caught.")
    # Failed registration remains stoppable and bounded.
    text = text.replace(
        "bool isRegistrationStatusLogged = false;",
        "bool isRegistrationStatusLogged = false;\n    unsigned registrationAttempts=0;",
    )
    text = text.replace(
        "} while( API_SUCCESS != result );",
        "} while( API_SUCCESS != result && !g_shutdown_requested && ++registrationAttempts<6 );",
    )
    daemon.write_text(text)
    cmake = project / "CMakeLists.txt"
    text = cmake.read_text()
    text = re.sub(r"set\(AGENT_DES[^\n]*", 'set(AGENT_DES "Fault Location Agent")', text)
    text = re.sub(
        r"set\(PROJECTGUID[^\n]*",
        f"set(PROJECTGUID {uuid.uuid5(uuid.NAMESPACE_URL, 'local.di/FaultLocationAgent')})",
        text,
    )
    text = re.sub(
        r"^.*target_compile_definitions\(FaultLocationAgent_Daemon PRIVATE FASTSIM=ON.*\n",
        "",
        text,
        flags=re.M,
    )
    for key, value in [("MAJORVERSION", "0"), ("MINORVERSION", "1"), ("BUILDNUMBER", "0")]:
        text = re.sub(r"set\(" + key + r' "[^"]*"\)', f'set({key} "{value}")', text)
    marker = "# Fault-location maintained sources"
    if marker in text:
        text = text.split(marker)[0]
    text += (
        "\n"
        + marker
        + "\ntarget_sources(FaultLocationAgent_Daemon PRIVATE src/fault_core.cpp src/fault_wire.cpp)\nif(NODAEMON)\n  target_sources(FaultLocationAgent_Daemon PRIVATE src/replay_io.cpp)\nendif()\n"
    )
    cmake.write_text(text)
    info = project / "info.sh"
    text = info.read_text()
    for key, value in [("MAJORVERSION", "0"), ("MINORVERSION", "1"), ("BUILDNUMBER", "0")]:
        text = re.sub("^" + key + r"=.*$", key + "=" + value, text, flags=re.M)
    # The six-channel port targets HW 4.2 polyphase only. These are the SDK
    # 3.0.0 release-table versions, not the older architecture-table examples.
    text = re.sub(r"^MINMETERVERSION=.*$", 'MINMETERVERSION="50.10.312.2"', text, flags=re.M)
    text = re.sub(r"^INSTALLON=.*$", "INSTALLON='GENX_PP=50.10.312.2,2.0.581.0'", text, flags=re.M)
    info.write_text(text)
    defaults = {
        "start": "0",
        "end": "-1",
        "manual_onset": "-1",
        "eta_i": "0.5",
        "eta_u": "0.85",
        "pre_seconds": "0.05",
        "post_seconds": "0.15",
        "fault_loop": "AUTO",
        "line_length_km": "50",
        "nominal_voltage_kv": "110",
        "base_power_mva": "100",
        "r1_ohm_km": "0.1",
        "x1_ohm_km": "0.4",
        "r0_ohm_km": "-1",
        "x0_ohm_km": "-1",
        "line_parameters_verified": "0",
        "settle_cycles": "1",
        "average_cycles": "3",
        "min_current_ka": "0.001",
        "fault_current_fraction": "0.8",
        "max_cycle_spread": "0.5",
        "known_distance_km": "-1",
    }
    for extension in ["xml", "desc.xml"]:
        path = project / f"config/etc/agents.d/23020000.23030000.{extension}"
        tree = ET.parse(path)
        folders = tree.getroot().find("folders")
        for folder in list(folders):
            if folder.get("name") != "General":
                folders.remove(folder)
        folder = ET.SubElement(folders, "folder", name="FaultLocation")
        for key, value in defaults.items():
            attributes = {"name": key, "value": value}
            if extension == "desc.xml":
                attributes.update(type="string", display=key, min="1", max="64")
            ET.SubElement(folder, "parameter", attributes)
        ET.indent(tree)
        tree.write(path, encoding="utf-8", xml_declaration=True)
    # No RMS subscription is used by this offline feature.
    policy = project / "config/PolicyFile.xml"
    text = policy.read_text()
    text = re.sub(r"<Metrology>.*?</Metrology>", "", text, flags=re.S)
    text = text.replace('<Permission name="Data:Subscription:Metrology:Partial"/>', "")
    policy.write_text(text)
    (project / "tools").mkdir(exist_ok=True)
    for name in ["host_session.py", "host_replay.py"]:
        shutil.copy2(source / "tools" / name, project / "tools" / name)
    for name in ["host_replay.sh", "host_run.sh"]:
        (project / name).write_text(
            '#!/bin/bash\nset -euo pipefail\ncd -- "$(dirname -- "$0")"\nexec python3 tools/host_replay.py "$@"\n'
        )
        (project / name).chmod(0o755)
    fixtures = source.parent / "runtime/di-agent/fixtures"
    if fixtures.exists():
        (project / "fixtures").mkdir(exist_ok=True)
        for name in ["1A_val1.wave", "1A_val1.json"]:
            shutil.copy2(fixtures / name, project / "fixtures" / name)
        manifest = json.loads((fixtures / "manifest.json").read_text())
        manifest["cases"] = [case for case in manifest["cases"] if case["name"] == "1A_val1"]
        (project / "fixtures/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ["README.md", "HARDWARE_TARGET.md", "hardware_profile.json"]:
        shutil.copy2(source / name, project / name)
    (project / "changelog.txt").write_text(
        "0.1.0: Native offline fault inception, sequence classification, reactance distance, and correlated DI alarm/data outputs. WSL waveform replay validated; live meter acquisition and resource qualification remain pending.\n"
    )
    subprocess.run(["chown", "-R", "fgonzales:fgonzales", str(project)], check=True)
    print("Installed maintained sources and configuration into", project)


if __name__ == "__main__":
    main()
