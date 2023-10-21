## 排序
**1、堆排序**
- https://github.com/PegasusWang/python_data_structures_and_algorithms/issues/16

**2、快速排序**

**3、归并排序**
```python
def sort(nums, l, r):
    if l >= r:
        return 

    mid = (l + r) // 2

    sort(nums, l, mid) # 对左边子数组进行排序
    sort(nums, mid+1, r) # 对右边子数组进行排序

    merge(nums, l, mid, r) # 合并排序好的左右子数组

def merge(nums, l, mid, r):
    for i in range(l, r+1):
        temp[i] = nums[i]

    i, j = l, mid+1
    p = l
    while p <= r:
        if i == mid+1:
            nums[p] = temp[j]
            j += 1
        elif j == r+1:
            nums[p] = temp[i]
            i += 1
        elif temp[i] > temp[j]:
            nums[p] = temp[j]
            j += 1
        else:
            nums[p] = temp[i]
            i += 1
        
        p += 1

nums = [5,2,3,1]
l, r = 0, len(nums)-1
temp = nums.copy()
sort(nums, l, r)
return nums
```
- 912 排序数组
- 315 计算右侧小于当前元素的个数


## 数组
**1、前缀和数组**

一维前缀和数组公式：`preSum[i] = preSum[i-1] + nums[i-1]`，表示数组nums中前 i 个元素的和
```python
nums = [3, 8, 7, 9, 4, 6, 5, 2, 0, 1]
preSum = [0]*(len(nums)+1)
for i in range(1, len(preSum)):
    preSum[i] = preSum[i-1] + nums[i-1]
```

二维前缀和数组公式：`preSum[i][j] = preSum[i-1][j] + preSum[i][j-1] - preSum[i-1][j-1] + nums[i-1][j-1]`，表示二维数组 nums 中从`(0,0)`到`(i-1,j-1)`之间的矩阵元素之和
```python
n, m = 4, 5
nums = [ [random.random() for c in range(m) ] for r in range(n)]
preSum = [ [0]*(m+1) for r in range(n+1)]
for r in range(1, n+1):
    for c in range(1, m+1):
        preSum[r][c] = preSum[r-1][c] + preSum[r][c-1] - preSum[r-1][c-1] + nums[r-1][c-1]
```
相关题目：
- 303 区域和检索 - 数组不可变
- 304 ⼆维区域和检索 - 矩阵不可变
- 1314 矩阵区域和

**2、差分数组**
差分数组公式：`diff[i] = nums[i] - nums[i-1]; nums[i] = nums[i-1] + diff[i]`
```python
nums = [3, 8, 7, 9, 4, 6, 5, 2, 0, 1]
diff = [0] * len(nums)
diff[0] = nums[0]
for i in range(1, len(nums)):
    diff[i] = nums[i] - nums[i-1]
```
相关题目：
- 370 区间加法
- 1109 航班预订统计
- 1094 拼车

**3、二分查找**




## 链表
**1、双指针**
相关题目：
- 5 最⻓回⽂⼦串
- 26 删除有序数组中的重复项
- 27 移除元素
- 283 移动零
- 19 删除链表的倒数第 N 个结点
- 21 合并两个有序链表
- 23 合并 K 个升序链表
- 83 删除排序链表中的重复元素
- 86 分隔链表
- 141 环形链表
- 142 环形链表 II
- 160 相交链表 
- 876 链表的中间结点
- 870 优势洗牌

**2、滑动窗口**
```python
# 记录窗⼝中的数据
window = {} # char: int
left = right = 0
while right < len(nums):
    c = nums[right]
    # 增⼤窗⼝
    right += 1 
    # 进⾏窗⼝内数据的⼀系列更新
    window.add(nums[right])
    ...
    
    # debug 输出的位置
    print(window, left, right)


    while left < right and window needs shrink:
        # d 是待移出窗⼝的字符
        d = nums[left]
        # 缩⼩窗⼝
        left += 1
        # 进⾏窗⼝内数据的⼀系列更新
        window.remove(nums[left])
        ...
```
相关题目：
- 76 最⼩覆盖⼦串
- 567 字符串的排列
- 438 找到字符串中所有字母异位词
- 3 无重复字符的最长子串

**3、反转单链**
- 206 反转链表
- 92 反转链表 II

**4、括号问题**
- 20 有效的括号
- 921 使括号有效的最少添加
- 1541 平衡括号字符串的最少插入次数


## 栈和队列
**1、单调栈**
常用于数组中求 下一个最大元素 等问题
```python
nums = [1,2,2,,3,4,5,6]
stack = []
res = []
for i in range(len(nums)-1, -1, -1)
    while stack and stack[-1] <= nums[i]:
        stack.pop()
    
    res[i] = stack[-1] if stack else -1
    stack.append(nums[i])
```
- 496 下一个更大元素 I
- 503 下一个更大元素 II
- 739 每日温度

**2、单调队列**
常用于滑动窗口中求最大值，下面是一个单调递减队列的定义
```python
class MonotonicQueue:
    def __init__(self):
        self.data = []
    
    def push(self, num):
        # 元素入队，注意保持队列的单调递减
        while self.data and self.data[-1] < num:
            self.data.pop()
        self.data.append(num)
    
    def pop(self, num):
        # 对头元素出队
        if num == self.data[0]:
            self.data.remove(num)
    
    def max(self, max):
        # 返回队列中的最大值
        return self.data[0]
```
- 239 滑动窗口最大值

**3、字符串去重**
- 316 去除重复字母
- 1081 不同字符的最小子序列

## 数据结构设计
**1、LRU(Least Recently Used) 算法**
常用的缓存策略，优先删除最久没有使用过的元素。底层实现为，双向链表和哈希表，哈希表存储的是 key 到 链表节点的映射。
```python
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key: int) -> int:
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:

        if key in self.data:
            self.data.move_to_end(key)
        else:
            if len(self.data) == self.capacity:
                self.data.popitem(last=False)

        self.data[key] = value
```

**2、LFU(Least Frequently Used) 算法算法**
常用的缓存策略，优先删除使用频率最少的元素，底层实现：1）key 到 频率的映射；2）频率到 key 的映射，注意，同一个频率可能有多个key，多个 key 之间有时间先后关系；3）记录最小频率；4）字典，存储数据；
```python
from typing import Optional, List
from collections import OrderedDict, defaultdict

class LFUCache:

    def __init__(self, capacity: int):
        self.k2freq = {}
        self.freq2k = defaultdict(OrderedDict)
        self.cache = {}
        self.capacity = capacity
        self.minfreq = 0


    def get(self, key: int) -> int:
        if key in self.cache:
            self.makeRecently(key)
            return self.cache[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key] = value
            self.makeRecently(key)
            return
        
        if len(self.cache) == self.capacity:
            self.deleteKey()
        
        self.cache[key] = value
        self.k2freq[key] = 1
        self.freq2k[1][key] = None
        self.minfreq = 1
        
    
    def makeRecently(self, key):
        freq = self.k2freq[key]
        self.k2freq[key] = freq + 1
        self.freq2k[freq+1][key] = None
        self.freq2k[freq].pop(key)
        if not self.freq2k[freq]:
            self.freq2k.pop(freq)
            if self.minfreq == freq:
                self.minfreq = freq + 1
    
    def deleteKey(self):
        key, _ = self.freq2k[self.minfreq].popitem(last=False)
        if not self.freq2k[self.minfreq]:
            self.freq2k.pop(self.minfreq)
        self.k2freq.pop(key)
        self.cache.pop(key)
```
- 460 LFU 缓存

**3、哈希数组**
- 380 O(1) 时间插入、删除和获取随机元素
- 710 黑名单中的随机数

**4、中位数**
- 295 数据流的中位数

**5、计算器**
```python
class Solution:
    def calculate(self, s: str) -> int:
        
        def helper(s: List):
            stack = []
            sign = "+"
            num = 0
            
            while len(s) > 0:
                c = s.pop(0)

                if c.isdigit():
                    num = num * 10 + int(c)
                
                if c == "(":
                    num = helper(s)

                if (not c.isdigit() and c != " ")  or len(s) == 0:
                    if sign == "+":
                        stack.append(num)
                    elif sign == "-":
                        stack.append(-num)
                    elif sign == "*":
                        stack[-1] = stack[-1] * num
                    elif sign == "/":
                        stack[-1] = int(stack[-1] / num)
                    
                    sign = c
                    num = 0
            
                if c == ")":
                    break
            
            return sum(stack)
        
        return helper(list(s))
```
- 224
- 227
- 772

## 树
- 104 ⼆叉树的最⼤深度
- 543 二叉树中的最大路径和
- 515 在每个树行中找最大值
- 226 翻转⼆叉树
- 116 填充每个节点的下一个右侧节点指针
- 114 二叉树展开为链表
- 654 最大二叉树
- 105 从前序与中序遍历序列构造二叉树
- 106 从中序与后序遍历序列构造二叉树
- 889 根据前序和后序遍历构造二叉树
- 297 二叉树的序列化与反序列化
- 652 寻找重复的子树
- 315 计算右侧小于当前元素的个数
- 493 翻转对
- 327 区间和的个数
- 230 二叉搜索树中第K小的元素
- 538 把二叉搜索树转换为累加树

## 二叉搜索树
- 98 验证⼆叉搜索树
- 700 二叉搜索树中的搜索
- 96 不同的二叉搜索树
- 95 不同的二叉搜索树 II
- 1373 二叉搜索子树的最大键值和









## 图论
## 回溯
## 动态规划

**1、求最长递增子序列**，假设数组为 nums，注意，子序列是非连续的。

dp 函数定义：dp(i) 表示以 nums[i] 这个数结尾的最⻓递增⼦序列的⻓度

状态转移方程：
```python
dp(i) = max(
    dp(i), 
    dp(j)+1 where j < i and nums[j] < nums[i]
)
```

**2、求最⼤的⼦数组和**，假设数组为 nums， 注意，子数组是连续

dp 函数定义：以 nums[i] 为结尾的「最⼤⼦数组和」为 dp(i)

状态转移方程：
```python
dp(i) = 
    if dp(i-1) + nums[i] < 0:
        max(
            nums[i],
            dp(i-1) + nums[i]
        )
    else:
        dp(i-1) + nums[i]
```

**3、求编辑距离**，给定两个单词 s，t，求将 s 转换为 t 的最小操作数（插入，删除，替换）

dp 函数定义：`dp(s, i, t, j)`  表示将 s[0: i+1] 转换为 t[0: j+1] 的最小操作数，注意索引含头不含尾。

状态转移方程：
```
dp(s, i, t, j) = 
    if s[i] == s[j]:
        dp(s, i-1, t, j-1)
    else:
        max(
            dp(s, i, t, j-1) + 1, # s 在 i 位置后面进行插入操作，
            dp(s, i-1, t, j) + 1, # 删除 s[i],
            dp(s, i-1, t, j-1) + 1, # s[i] 替换为 s[j]
        )
```

**4、求最⻓公共⼦序列**，注意子序列是非连续的

dp 函数定义：`dp(s1, i, s2, j)`  表示求 s1[0: i+1] 和 s2[0: j+1] 的最长公共子序列，注意索引含头不含尾。

状态转移方程：
```
dp(s1, i, s2, j) = 
    if s[i] == s[j]:
        dp(s1, i-1, s2, j-1) + 1
    else:
        max(
            dp(s1, i-1, s2, j),
            dp(s1, i, s2, j-1)
        )
```

**5、0-1背包问题**，问题描述：给定载重为 W 的背包和 N 个物品，物品有重量和价值两个属性，其中第 i 个物品的重量表示 `w[i]`，价值表示为 `v[i]`，求背包可装满物品的最大价值。

dp 函数定义：dp(i, j) 表示对于前 i 个物品，当前背包容量为 j 时，背包可装的最大价值，

状态转移方程：
```
dp(i, j) =  
    if j - w[i] < 0: # 无法装入第 i 个物品
        dp(i-1, j)
    else:
        max(
            dp(i-1, j), # 第 i 个物品不装进背包
            dp(i-1, j-w[i]) + v[i], # 第 i 个物品装进背包
        )

base case:
dp(0, ...) = 0，对于前 0 个物品，即没有物品可选，背包价值为0
dp(..., 0) = 0，背包载重为0，背包价值为0
```

**6、完全背包问题**，问题描述：给定不同⾯额的硬币 coins 和⼀个总⾦额 amount，假设
每⼀种⾯额的硬币有⽆限个，求可以凑成总⾦额的硬币组合数

dp 函数定义：dp(i, j) 表示只使用前 i 个面额的硬币，可以凑成总金额 j 的硬币组合数

状态转移方程：
```
dp(i, j) = 
    if j - coins[i] < 0: # 无法使用面额为coins[i]的硬币
        dp(i-1, j)
    else:
        dp(i-1, j) + dp(i, j-coins[i])

base case:
dp(0, j) = 0，没有可用的硬币，凑不出总金额 j，所以硬币组合数为0
dp(i, 0) = 1，要凑出的总金额为 0，即不做任何选择，空选择也算一种硬币组合。
```

**7、⼦集背包问题**，问题描述：输⼊⼀个只包含正整数的非空数组 nums，判断这个数组是否可以被分割成两个⼦集，使得两个⼦集的元素和相等。

dp 函数定义：dp(i, j) 表示使用前 i 个数字，其加和恰好等于j的可行性，可行返回true，否则返回false

状态转移方程：
```
dp(i, j) = 
    if j - nums[i] < 0: # 不能使用第 i 个数字
        dp(i-1, j)

    else:
        dp(i-1, j) || dp(i-1, j-nums[i])
base case:
dp(i, 0) = True，使用前 i 个数字，其加和恰好为0，不选择任何数字，即可
dp(0, j) = False，不使用任何数字，其加和恰好为j，无解
```

**8、股票问题**，问题描述：给定一个数组 prices，`prices[i]` 表示一只股票在第 i 天的股价，你最多可以完成 K 笔交易，一次买卖表示一笔交易，求你能获得的最大利润。你不能同时参与多笔交易，必须在卖出股票后才能进行下一次买入

dp 数组定义：
```python
dp[i][k][0 or 1], 0<=i<=n-1, 1<=k<=K，i 表示天数，k 表示交易次数，0，1表示持股状态，0 表示不持有，1 表示持有
```
>数组含义：第 i 天，最大交易次数为k，手上持股或不持股时的最大利润。我们最终要求的就是`dp[i][k][0]`，因为`dp[i][k][1]`代表到最后⼀天⼿上还持有股票，肯定要比前者（股票已经卖出了）的利润的小。
**注意**：k=1 或 k=正无穷时，可以忽略k，此时dp数组可以简化为 `dp[i][0 or 1]`

状态转移方程：
```python
dp[i][k][0] = max(
    dp[i-1][k][0], # 第 i-1 天没有持股，今天不交易
    dp[i-1][k][1] + prices[i] # 第 i-1 天有持股，今天卖出
)
dp[i][k][1] = max(
    dp[i-1][k][1], # 第 i-1 天有持股，今天不交易继续持有
    dp[i-1][k-1][0] - prices[i] # 第 i-1 天没有持股，今天买入，交易次数+1
)
# 一次买卖表示一笔交易，在买入时，开始计数交易次数

# base case:
dp[-1][...][0] = 0, # 交易没有开始，手上没有持股，所以利润为 0
dp[...][0][0] = 0, # 最大交易次数为0，手上没有持股，所以利润为 0
dp[-1][...][1] = "-inf", # 交易没有开始，手上有持股，所以利润为负无穷（要求最大值）
dp[...][0][1] = "-inf", # 最大交易次数为0，手上有持股，所以利润为负无穷（要求最大值）

# 当 k=1 或 k=正无穷时，将 dp 数组中 k 这个维度去掉即可
```
<details open="true">
<summary>相关题目</summary>

- 121 题「买卖股票的最佳时机」
- 122 题「买卖股票的最佳时机 II」
- 123 题「买卖股票的最佳时机 III」
- 188 题「买卖股票的最佳时机 IV」
- 309 题「最佳买卖股票时机含冷冻期」
- 714 题「买卖股票的最佳时机含⼿续费」 
</details>

**9、打家劫舍问题**，问题描述：街上有⼀排房屋，⽤⼀个包含⾮负整数的数组 nums 表示，每个元素 `nums[i]` 代表第 i 间房⼦中的现⾦数额。现在你是⼀名专业⼩偷，你希望尽可能多的盗窃这些房⼦中的现⾦，但是，相邻的房⼦不能被同时盗窃，计算在不触动报警器的前提下，最多能够盗窃多少现⾦呢

dp 函数定义：dp(i) 表示从第 i 间房子开始偷，最多能偷到的钱

状态转移方程：
```python
dp(i) = max(
    dp(i+1), # 不偷当前房子，去偷下一家
    nums[i] + dp(i+2) # 偷当前房子，不偷下一家
)
base case：
dp(i) = 0, i >= nums.length 
```
<details open="true">
<summary>相关题目</summary>

- 198 打家劫舍 
- 213 打家劫舍 II 
- 337 打家劫舍 III
</details>

**10、博弈问题**：
问题描述：
>给你一个整数数组 nums 。玩家 1 和玩家 2 基于这个数组设计了一个游戏。
>
>玩家 1 和玩家 2 轮流进行自己的回合，玩家 1 先手。开始时，两个玩家的初始分值都是 0 。每一回合，玩家从数组的任意一端取一个数字（即，`nums[0]` 或 `nums[nums.length - 1]`），取到的数字将会从数组中移除（数组长度减 1 ）。玩家选中的数字将会加到他的得分上。当数组中没有剩余数字可取时，游戏结束。
>
>如果玩家 1 能成为赢家，返回 true 。如果两个玩家得分相等，同样认为玩家 1 是游戏的赢家，也返回 true 。你可以假设每个玩家的玩法都会使他的分数最大化。

dp 数组定义：
>定义二维的 dp 数组，每个元素都是一个元组，形如：(fir, sec)，dp 数组的含义如下：
>
>`dp[i][j].fir = x` 表示，对于 `nums[i...j]`，先⼿能获得的最⾼分数为 x。
>
>`dp[i][j].sec = y` 表示，对于 `nums[i...j]`，后⼿能获得的最⾼分数为 y。

状态转移方程：
```python
状态有三个：开始的索引 i，结束的索引 j，当前轮到的⼈
dp[i][j][fir or sec], 
0 <= i < len(nums),
i <= j < len(nums), 

# 我作为先⼿
dp[i][j].fir = max(
    nums[i] + dp[i+1][j].sec, # 先手拿 nums 最左边元素，然后变为后手
    nums[j] + dp[i][j-1].sec  # 先手拿 nums 最右边元素，然后变为后手
)
# 我作为后⼿
dp[i][j].sec = max(
    dp[i+1][j].fir, # 先⼿选择了最左边元素，给我剩下了dp[i+1...j]，此时轮到我，我变成了先⼿
    dp[i][j-1].fir  # 先⼿选择了最右边元素，给我剩下了dp[i...j-1]，此时轮到我，我变成了先⼿
)
base case: i 等于 j，即nums只剩下一个元素时，
dp[i][j].fir = nums[i], i=j, # 先手拿走最后一个元素
dp[i][j].sec = 0, i=j, # 后手没有元素可拿
```
<details open="true">
<summary>相关题目</summary>

- 486 预测赢家
- 877 ⽯⼦游戏
</details>

**11、最⼩路径和**
问题描述：
>现在给你输⼊⼀个⼆维数组 grid，其中的元素都是⾮负整数，现在你站在左上⻆，只能向右或者向下移动，需要到达右下⻆。现在请你计算，经过的路径和最⼩是多少？

dp 函数定义：dp(i,j)，表示从起始点 (0,0) 开始到位置 (i,j) 的最小路径和

状态转移方程：
```python
dp(i,j) = min(
    dp(i-1, j) + grid[i][j], # 向下走，达到当前位置
    dp(i, j-1) + grid[i][j]  # 向右走，达到当前位置
)
base case:
dp(0, 0) = grid[0][0]
```
相关题目：
- 64 最⼩路径和


**12、地下城游戏**
问题描述：
>输⼊⼀个存储着整数的⼆维数组 grid，如果 `grid[i][j] > 0`，说明这个格⼦装着⾎瓶，经过它可以增加对应的⽣命值；如果 `grid[i][j] == 0`，则这是⼀个空格⼦，经过它不会发⽣任何事情；如果 `grid[i][j] < 0`，说明这个格⼦有怪物，经过它会损失对应的⽣命值。现在你是⼀名骑⼠，将会出现在最左上⻆，公主被困在最右下⻆，你只能向右和向下移动，请问你初始⾄少需要多少⽣命值才能成功救出公主？换句话说，就是问你⾄少需要多少初始⽣命值，能够让骑⼠从最左上⻆移动到最右下⻆，且任何时候⽣命值都要⼤于 0。

dp 函数定义：dp(i,j) 表示从位置 (i,j) 开始到右下角终点，骑士最少需要多少生命值

状态转移方程：
```python
res = min(
    dp(i+1, j),
    dp(i, j+1)
    ) - grid[i][j] 
dp(i, j) = 1 if res <= 0 else res

base case: i=m-1, j=n-1
dp(i, j) = 1 if grid[i][j] >= 0 else -grid[i][j] + 1
```
相关题目：
- 174 题「地下城游戏」


**13、⾼楼扔鸡蛋**
问题描述：
>给你 k 枚相同的鸡蛋，并可以使用一栋从第 1 层到第 n 层共有 n 层楼的建筑。
>
>已知存在楼层 f ，满足 `0 <= f <= n` ，任何从 高于 f 的楼层落下的鸡蛋都会碎，从 f 楼层或比它低的楼层落下的鸡蛋都不会破。
>
>每次操作，你可以取一枚鸡蛋把它从任一楼层 x 扔下（满足 `1 <= x <= n`）。如果鸡蛋碎了，你就不能再次使用它。如果没碎，则可以在之后的操作中 重复使用 这枚鸡蛋。
>
>请你计算确定 f 确切的值 的 最小操作次数 是多少？

解法一：

dp 函数定义：`dp(k,n)` 表示使用k枚鸡蛋，确定n层楼，需要的的最小操作次数，其中 k 为鸡蛋数，n 为楼层数

状态转移方程：
```math
dp(k,n) = 1 + \displaystyle \min_{1 \leqslant x \leqslant n}\left ( max(dp(k,n-x),dp(k-1,x-1)) \right )
```
其中 x 表示从第 x 楼扔鸡蛋
>`dp(k,n-x)` 表示如果鸡蛋不碎，则答案只能在上方的 n-x 层楼
>`dp(k-1,x-1)` 表示如果鸡蛋碎了，则答案只能在下方的 x-1 层楼
> base case：`dp(..., 0) = 0`, 表示只有 0 层楼时，最小操作数为0；`dp(1, x) = x`, 表示只有 1 枚鸡蛋时，确定 x 层楼的最小操作为 x，即从第 1 楼开始逐层向上扫描。

解法二：

dp 函数定义：`f(t,k)` 表示当前有 k 个鸡蛋，最多做 t 次操作，我们能确定的最大楼层数。那么现在的问题就是，要找到最小的操作次数 t，满足 `f(t,k) >= n`, 

状态转移方程：
```math
f(t,k) = 1 + f(t-1,k) + f(t-1,k-1)
```
方程含义，当某次扔出一枚鸡蛋时，状态的变化如下：
- `f(t-1,k)` 表示如果鸡蛋没碎，那么在这一层的上方可以有 `f(t-1,k)` 层，操作数减少一次；
- `f(t-1,k-1)` 表示如果鸡蛋碎了，那么在这一层的下方可以有 `f(t-1,k-1)` 层，操作数减少一次，同时鸡蛋数量减少一枚；
- 最后总的楼层数就是：`1 + f(t-1,k) + f(t-1,k-1)`
- base case: 
  - `f(0,...) = f(..., 0) = 0`，操作数为 0 或 鸡蛋数为 0 时，确定层楼数为 0；
  - t 的范围：`1<=t<=n`，t 肯定不会超过最大楼层数；


相关题目：
- 887 鸡蛋掉落









