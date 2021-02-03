package solution400;

import java.util.Comparator;
import java.util.TreeMap;

public class Solution480 {

    /**
     * 480. 滑动窗口中位数
     * @param nums
     * @param k
     * @return
     */
    public double[] medianSlidingWindow(int[] nums, int k) {
        // TODO 数字超出21亿
        TreeMap<Integer, Integer> m1 =  new TreeMap<>(Comparator.reverseOrder());
        TreeMap<Integer, Integer> m2 = new TreeMap<>();
        int size1 = 0;
        int size2 = 0;
        double[] res = new double[nums.length - k + 1];
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            // 插入操作
            if (size1 == 0) {
                insert(m1, nums[i]);
                size1++;
            } else {
                if (nums[i] <= getFirst(m1)) {
                    insert(m1, nums[i]);
                    size1++;
                } else {
                    insert(m2, nums[i]);
                    size2++;
                }
            }
            if (i >= k) {
                // 删除操作
                if (m1.containsKey(nums[i - k])) {
                    delete(m1, nums[i - k]);
                    size1--;
                } else {
                    delete(m2, nums[i - k]);
                    size2--;
                }
            }
            if (i >= k - 1) {
                // 获得中位数，先调整两个堆的大小关系
                while (size1 - size2 > 1) {
                    int a = getFirst(m1);
                    insert(m2, a);
                    size2++;
                    delete(m1, a);
                    size1--;
                }
                while (size1 - size2 < 0) {
                    int a = getFirst(m2);
                    insert(m1, a);
                    size1++;
                    delete(m2, a);
                    size2--;
                }
                if (k % 2 == 0) {
                    res[index++] = (getFirst(m1) + getFirst(m2)) / 2d;
                } else {
                    res[index++] = getFirst(m1);
                }
            }
        }
        return res;
    }

    private int getFirst(TreeMap<Integer, Integer> map) {
        return map.keySet().iterator().next();
    }

    private void insert(TreeMap<Integer, Integer> map, int x) {
        map.put(x, map.getOrDefault(x, 0) + 1);
    }

    private void delete(TreeMap<Integer, Integer> map, int x) {
        if (map.get(x) == 1) {
            map.remove(x);
        } else {
            map.put(x, map.get(x) - 1);
        }
    }

    public static void main(String[] args) {
        Solution480 solution = new Solution480();
        int[] nums = {1, 3, -1, -3, 5, 3, 6, 7};
        double[] ans = solution.medianSlidingWindow(nums, 3);
        for (double d : ans) {
            System.out.println(d);
        }
    }
}
