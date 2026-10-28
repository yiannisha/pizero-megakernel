// Development-only launch tracing. No GPU performance counters are requested.
#include <cupti.h>
#include <atomic>
#include <cstring>
#include <mutex>

struct LaunchTrace {
    CUpti_SubscriberHandle subscriber{};
    std::atomic<int> count{0};
    std::mutex mutex;
    char symbol[2048]{};
};
static void CUPTIAPI callback(void* userdata,CUpti_CallbackDomain domain,CUpti_CallbackId,const void* payload){
    if(domain!=CUPTI_CB_DOMAIN_DRIVER_API)return;
    auto* data=static_cast<const CUpti_CallbackData*>(payload);
    if(data->callbackSite!=CUPTI_API_ENTER || !data->functionName)return;
    // Also reject extra cooperative/legacy launches, host functions and graphs
    // in a scope expected to contain exactly one ordinary native kernel.
    if(std::strncmp(data->functionName,"cuLaunch",8) && std::strncmp(data->functionName,"cuGraphLaunch",13))return;
    auto* trace=static_cast<LaunchTrace*>(userdata);
    trace->count.fetch_add(1,std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(trace->mutex);
    if(data->symbolName){std::strncpy(trace->symbol,data->symbolName,sizeof(trace->symbol)-1);trace->symbol[sizeof(trace->symbol)-1]=0;}
}
extern "C" int begin_launch_trace(void** handle){
    if(!handle)return CUPTI_ERROR_INVALID_PARAMETER;
    *handle=nullptr;auto* trace=new LaunchTrace;
    auto error=cuptiSubscribe(&trace->subscriber,callback,trace);
    if(error==CUPTI_SUCCESS){
        error=cuptiEnableDomain(1,trace->subscriber,CUPTI_CB_DOMAIN_DRIVER_API);
        if(error!=CUPTI_SUCCESS)cuptiUnsubscribe(trace->subscriber);
    }
    if(error==CUPTI_SUCCESS)*handle=trace;else delete trace;
    return error;
}
extern "C" int end_launch_trace(void* handle,int* count,char* symbol,int capacity){
    if(!handle||!count||!symbol||capacity<1)return CUPTI_ERROR_INVALID_PARAMETER;
    auto* trace=static_cast<LaunchTrace*>(handle);
    auto error=cuptiEnableDomain(0,trace->subscriber,CUPTI_CB_DOMAIN_DRIVER_API);
    auto unsubscribed=cuptiUnsubscribe(trace->subscriber);
    if(unsubscribed!=CUPTI_SUCCESS)return unsubscribed;
    *count=trace->count.load();std::strncpy(symbol,trace->symbol,capacity-1);symbol[capacity-1]=0;
    delete trace;return error;
}
extern "C" const char* launch_trace_error(int code){
    const char* message=nullptr;cuptiGetResultString(static_cast<CUptiResult>(code),&message);return message;
}
