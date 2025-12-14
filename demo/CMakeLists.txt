# ============================================
# 主可执行文件配置
# ============================================

# 创建Llama模型推理主程序
add_executable(llama_infer main.cpp)

# 设置链接库搜索路径
target_link_directories(llama_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)

# 链接主库
target_link_libraries(llama_infer llama)

# ============================================
# LLAMA3模型支持配置
# ============================================

if(LLAMA3_SUPPORT)
    message(STATUS "链接 LLAMA3 支持库")
    
    # 查找并链接LLAMA3所需依赖
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    
    # 链接LLAMA3特定库
    target_link_libraries(llama_infer 
        absl::base 
        re2::re2 
        nlohmann_json::nlohmann_json
    )
endif()

# ============================================
# 设置CUDA可分离编译属性
# ============================================

set_target_properties(llama_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================
# QWEN2模型支持配置
# ============================================

if(QWEN2_SUPPORT)
    message(STATUS "链接 QWEN2 支持库")
    
    # 创建Qwen2模型推理程序
    add_executable(qwen_infer main_qwen.cpp)
    
    # 设置链接库搜索路径
    target_link_directories(qwen_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
    
    # 查找Qwen2所需依赖
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    
    # 链接库依赖
    target_link_libraries(qwen_infer 
        llama  # 主库
        absl::base 
        re2::re2 
        nlohmann_json::nlohmann_json
    )
    
    # 设置CUDA可分离编译属性
    set_target_properties(qwen_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
# ============================================
# QWEN3模型支持配置
# ============================================
elseif(QWEN3_SUPPORT)
    message(STATUS "链接 QWEN3 支持库")
    
    # 创建Qwen3模型推理程序
    add_executable(qwen3_infer main_qwen3.cpp)
    
    # 设置链接库搜索路径
    target_link_directories(qwen3_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
    
    # 查找Qwen3所需依赖
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    
    # 链接库依赖
    target_link_libraries(qwen3_infer 
        llama  # 主库
        absl::base 
        re2::re2 
        nlohmann_json::nlohmann_json
    )
    
    # 设置CUDA可分离编译属性
    set_target_properties(qwen3_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()