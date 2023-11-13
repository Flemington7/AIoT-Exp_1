import numpy as np

# 假设这是你的两个字符串数组
array1 = np.array(["hello", "world", "python"])
array2 = np.array([123, 456, 789])
array3 = array2.astype(str)

# 对应元素相拼接
result = np.char.add(array1, array3)

# 打印结果
print(array2)
# 输出：['hello123', 'world456', 'python789']
