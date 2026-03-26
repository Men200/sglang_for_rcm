#!/usr/bin/env python3
"""
测试任务管理系统中已注册的模型
"""
import requests
import json

BASE_URL = "http://wanqing-aigc-workflow-candidate.internal"
PULL_URL = BASE_URL + "/inner-api/v1/diffusion_inference/tasks/pull"

# 测试的模型列表
TEST_MODELS = [
    "wan2.1-t2v-14b-rcm",
    "wan2.1-t2v-14b",
    "wan2.2-t2v-a14b",
    "wan2.2-i2v-a14b",
    "qwen-image",
    "qwen-image-edit-2509",
    "z-image-turbo",
]

def test_model(model_name, stage="CANDIDATE"):
    """测试某个模型是否在任务管理系统中注册"""
    payload = {
        "workerId": "test-worker",
        "model": model_name,
        "stage": stage,
        "modelService": model_name,
    }
    
    try:
        response = requests.post(PULL_URL, json=payload, timeout=5)
        resp_data = response.json()
        
        print(f"\n【{model_name}】")
        print(f"  HTTP Status: {response.status_code}")
        print(f"  Response: {json.dumps(resp_data, indent=2, ensure_ascii=False)}")
        
        if resp_data.get("code") == 0:
            print(f"  ✅ 模型已注册")
            return True
        elif "workflow config not found" in resp_data.get("msg", ""):
            print(f"  ❌ 模型未注册或工作流配置不存在")
            return False
        elif resp_data.get("code") == 100201:
            print(f"  ⏳ 模型已注册，但暂无可用任务")
            return True
        else:
            print(f"  ⚠️ 未知状态")
            return None
            
    except Exception as e:
        print(f"\n【{model_name}】")
        print(f"  ❌ 错误: {e}")
        return False

if __name__ == "__main__":
    print(f"🔍 测试任务管理系统: {BASE_URL}")
    print(f"📍 API地址: {PULL_URL}")
    
    registered = []
    not_registered = []
    
    for model in TEST_MODELS:
        result = test_model(model)
        if result is True:
            registered.append(model)
        elif result is False:
            not_registered.append(model)
    
    print("\n" + "="*50)
    print("📊 测试总结：")
    if registered:
        print(f"\n✅ 已注册的模型 ({len(registered)}):")
        for m in registered:
            print(f"   - {m}")
    if not_registered:
        print(f"\n❌ 未注册的模型 ({len(not_registered)}):")
        for m in not_registered:
            print(f"   - {m}")
