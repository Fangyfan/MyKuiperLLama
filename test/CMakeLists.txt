# ============================================
# 包含上级目录配置
# ============================================

# 包含CUDA相关配置
include(../cmake/cuda.cmake)

# ============================================
# 查找外部依赖包
# ============================================

# 查找Google Test框架
find_package(GTest REQUIRED)

# 查找Google Logging库
find_package(glog REQUIRED)

# ============================================
# 外部链接库变量定义
# ============================================

set(link_ext_lib 
    glog::glog 
    GTest::gtest
)

# ============================================
# 收集测试源代码
# ============================================

# 收集所有测试文件
aux_source_directory(../test DIR_TEST)
# 收集CUDA相关测试文件
aux_source_directory(../test/test_cu DIR_TEST_CU)
# 收集操作符相关测试文件
aux_source_directory(../test/test_op DIR_TEST_OP)
# 收集模型相关测试文件
aux_source_directory(../test/test_model DIR_TEST_MODEL)
# 收集张量相关测试文件
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
# 收集优化相关测试文件
aux_source_directory(../test/optimized DIR_TEST_OPTIMIZED)

# ============================================
# 创建测试可执行文件
# ============================================

add_executable(test_llm 
    ${DIR_TEST}          # 通用测试
    ${DIR_TEST_CU}       # CUDA测试
    ${DIR_TEST_OP}       # 操作符测试
    ${DIR_TEST_OPTIMIZED} # 优化测试
    ${DIR_TEST_TENSOR}   # 张量测试
    ${DIR_TEST_MODEL}    # 模型测试
)

# ============================================
# 链接外部库依赖
# ============================================

target_link_libraries(test_llm ${link_ext_lib})

# ============================================
# 设置包含目录
# ============================================

# Google Logging头文件目录
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})

# Google Test头文件目录
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

# 项目主头文件目录
target_include_directories(test_llm PUBLIC ../kuiper/include)

# ============================================
# 设置链接目录
# ============================================

target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)

# ============================================
# 模型特定依赖配置
# ============================================

if(LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    message(STATUS "链接模型特定依赖 (LLAMA3/QWEN2/QWEN3)")
    
    # 查找模型特定依赖包
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    
    # 链接依赖到主库（注意：这里链接到llama库，不是test_llm）
    target_link_libraries(llama 
        absl::base 
        re2::re2 
        nlohmann_json::nlohmann_json
    )
endif()

# ============================================
# 链接主项目库
# ============================================

target_link_libraries(test_llm llama)

# ============================================
# 目标属性设置
# ============================================

# 设置工作目录为项目根目录（便于访问测试数据）
set_target_properties(test_llm PROPERTIES 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# 启用CUDA可分离编译
set_target_properties(test_llm PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)