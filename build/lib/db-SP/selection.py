import torch

list1 = torch.load("/root/chensiqi/Dual-balance/total_time_list_u8r1.pt")
list2 = torch.load("/root/chensiqi/Dual-balance/total_time_list_u4r2.pt")
list3 = torch.load("/root/chensiqi/Dual-balance/total_time_list_u2r4.pt")
list4 = torch.load("/root/chensiqi/Dual-balance/total_time_list_u1r8.pt")
# 假设有4个列表：list1, list2, list3, list4
# 每个列表有50个子列表，每个子列表有42个数

# 初始化计数器
count_list1 = 0
count_list2 = 0
count_list3 = 0
count_list4 = 0
total_min_sum = 0

# 计算各个列表的总和
total_sum_list1 = sum(sum(sublist) for sublist in list1)
total_sum_list2 = sum(sum(sublist) for sublist in list2)
total_sum_list3 = sum(sum(sublist) for sublist in list3)
total_sum_list4 = sum(sum(sublist) for sublist in list4)

# 遍历所有位置，计算最小值总和和统计选择次数
for i in range(50):  # 遍历50个子列表
    for j in range(40):  # 遍历每个子列表的42个位置
        # 获取当前位置四个列表的值
        val1 = list1[i][j]
        val2 = list2[i][j]
        val3 = list3[i][j]
        val4 = list4[i][j]
        
        # 找到最小值
        min_val = min(val1, val2, val3, val4)
        total_min_sum += min_val
        
        # 统计哪个列表提供了最小值
        # 如果有多个相同的最小值，我们只统计第一个遇到的
        if val1 == min_val:
            count_list1 += 1
        elif val2 == min_val:
            count_list2 += 1
        elif val3 == min_val:
            count_list3 += 1
        else:
            count_list4 += 1

# 输出结果
print(f"所有位置最小值的总和: {total_min_sum}")
print(f"从 list1 选择的个数: {count_list1}")
print(f"从 list2 选择的个数: {count_list2}")
print(f"从 list3 选择的个数: {count_list3}")
print(f"从 list4 选择的个数: {count_list4}")
print(f"list1 的总和: {total_sum_list1}")
print(f"list2 的总和: {total_sum_list2}")
print(f"list3 的总和: {total_sum_list3}")
print(f"list4 的总和: {total_sum_list4}")