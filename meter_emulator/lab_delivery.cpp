// Observe completion of actual SDK calls. No results or policy responses are
// substituted. This library is installed only in the isolated lab image.
#include <Itron/AgentApi.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
// This meter uses uClibc; avoid compiler TLS in a preloaded library.
static pthread_key_t depth_key;
static pthread_once_t depth_once = PTHREAD_ONCE_INIT;
static void make_depth_key() { pthread_key_create(&depth_key, 0); }
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static void record(uint32_t feature, const char *data, uint32_t length, bool event, bool alarm, int result) {
    if (feature != 0x23030000 && feature != 0x23030001) return;
    const char *dir = getenv("METER_LAB_DIR");
    if (!dir) dir = getenv("METER_LAB_METROLOGY_DIR");
    if (!dir) return;
    uint64_t hash=UINT64_C(14695981039346656037);
    for (uint32_t i=0;i<length;++i) { hash ^= static_cast<unsigned char>(data[i]); hash *= UINT64_C(1099511628211); }
    char path[1024]; snprintf(path,sizeof(path),"%s/sdk-completions.jsonl",dir);
    pthread_mutex_lock(&lock);
    FILE *file=fopen(path,"a");
    if(file) { fprintf(file,"{\"key\":\"%llx%c\",\"feature\":%u,\"bytes\":%u,\"event\":%s,\"alarm\":%s,\"return_code\":%d,\"utc\":%lld}\n",
        static_cast<unsigned long long>(hash),event?'e':'d',feature,length,event?"true":"false",alarm?"true":"false",result,static_cast<long long>(time(0))); fclose(file); }
    pthread_mutex_unlock(&lock);
}
#define CALL_REAL(name,args,feature,data,length,event,alarm) \
    auto real=reinterpret_cast<decltype(&name)>(dlsym(RTLD_NEXT,#name)); \
    if(!real) return static_cast<AgentApiReturnType>(-1); \
    pthread_once(&depth_once, make_depth_key); \
    uintptr_t depth=reinterpret_cast<uintptr_t>(pthread_getspecific(depth_key)); \
    pthread_setspecific(depth_key,reinterpret_cast<void *>(depth+1)); \
    auto result=real args; \
    pthread_setspecific(depth_key,reinterpret_cast<void *>(depth)); \
    if(!depth) record(feature,data,length,event,alarm,static_cast<int>(result)); return result
extern "C" {
AgentApiReturnType ApiWriteAgentDataForUpstreamSystem(uint32_t f,const char *d,uint32_t n) {
    CALL_REAL(ApiWriteAgentDataForUpstreamSystem,(f,d,n),f,d,n,false,false);
}
AgentApiReturnType ApiWriteAgentDataForUpstreamSystemWithTimestamp(uint32_t f,const char *d,uint32_t n,uint64_t t) {
    CALL_REAL(ApiWriteAgentDataForUpstreamSystemWithTimestamp,(f,d,n,t),f,d,n,false,false);
}
AgentApiReturnType ApiLogHistoryEvent(uint32_t e,const char *d,uint32_t n,uint32_t f,bool a) {
    CALL_REAL(ApiLogHistoryEvent,(e,d,n,f,a),f,d,n,true,a);
}
AgentApiReturnType ApiLogHistoryEventWithTimestamp(uint32_t e,const char *d,uint32_t n,uint32_t f,bool a,uint64_t t) {
    CALL_REAL(ApiLogHistoryEventWithTimestamp,(e,d,n,f,a,t),f,d,n,true,a);
}
AgentApiReturnType ApiLogHistoryAlarm(uint32_t e,const char *d,uint32_t n,uint32_t f) {
    CALL_REAL(ApiLogHistoryAlarm,(e,d,n,f),f,d,n,true,true);
}
AgentApiReturnType ApiLogHistoryAlarmWithTimestamp(uint32_t e,const char *d,uint32_t n,uint32_t f,uint64_t t) {
    CALL_REAL(ApiLogHistoryAlarmWithTimestamp,(e,d,n,f,t),f,d,n,true,true);
}
}
