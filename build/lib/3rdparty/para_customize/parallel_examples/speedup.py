import pandas as pd
import os

# 定义要处理的文件列表
file_list = ['paro_mask1.csv', 'paro_mask2.csv', 'sparge.csv']

for file_name in file_list:
    try:
        # 检查文件是否存在
        if not os.path.exists(file_name):
            print(f"文件 {file_name} 不存在，跳过...")
            continue
            
        # 读取CSV文件
        df = pd.read_csv(file_name)
        
        # 获取第一行的timing_sec和attention_time_s作为基准
        base_timing = df.loc[0, 'timing_sec']
        base_attention = df.loc[0, 'attention_time_s']
        
        # 计算比值，保留3位小数
        df['end_to_end_speedup'] = (base_timing / df['timing_sec']).round(3)
        df['attention_speedup'] = (base_attention / df['attention_time_s']).round(3)
        
        # 保存回原文件
        df.to_csv(file_name, index=False)
    
        
    except Exception as e:
        print(f"✗ {file_name} went wrong: {e}")

print("All files are finished.")